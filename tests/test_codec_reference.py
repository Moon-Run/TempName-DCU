"""CPU-only independent arithmetic and packet-contract checks."""
import ctypes
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from flux_dcu import reference as ref
from flux_dcu.codec import packet_bytes


def adjacent_fp32(value, direction):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + direction))[0]


class E4M3OracleTest(unittest.TestCase):
    def test_production_scalar_conversion_against_independent_oracle(self):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable for CPU arithmetic cross-check")
        # Compile the actual device scalar helper bodies on CPU. This validates
        # conversion arithmetic; it is NOT a HIP compiler or GPU execution test.
        source = (ROOT / "src/taco/codec.hip").read_text()
        body = source[source.index("__device__ __forceinline__ uint32_t bits_of"):
                      source.index("__device__ __forceinline__ void hadamard")]
        body = body.replace("__device__", "").replace("__forceinline__", "inline")
        host = ("#include <cmath>\n#include <cstdint>\n" + body +
                '\nextern "C" unsigned char encode_host(float x) { return to_e4m3(x); }\n' +
                'extern "C" float decode_host(unsigned char x) { return from_e4m3(x); }\n' +
                'extern "C" unsigned short bf16_host(float x) { return write_bf16(x); }\n')
        with tempfile.TemporaryDirectory(prefix="dcu-codec-cpu-") as folder:
            cpp, library = Path(folder) / "conversion.cc", Path(folder) / "conversion.so"
            cpp.write_text(host)
            subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(cpp),
                            "-o", str(library)], check=True, capture_output=True, text=True)
            native = ctypes.CDLL(str(library))
            native.encode_host.argtypes, native.encode_host.restype = [ctypes.c_float], ctypes.c_ubyte
            native.decode_host.argtypes, native.decode_host.restype = [ctypes.c_ubyte], ctypes.c_float
            native.bf16_host.argtypes, native.bf16_host.restype = [ctypes.c_float], ctypes.c_ushort
            values = [0.0, -0.0, 448., -448., 1e20, -1e20]
            for low in range(126):
                middle = (ref.e4m3_decode(low) + ref.e4m3_decode(low + 1)) / 2
                for x in (adjacent_fp32(middle, -1), middle, adjacent_fp32(middle, 1)):
                    values.extend((x, -x))
            rng = random.Random(71)
            values.extend(rng.uniform(-500, 500) for _ in range(1000))
            for value in values:
                self.assertEqual(native.encode_host(value), ref.e4m3_encode(value), value)
                self.assertEqual(native.bf16_host(value), ref.bf16_bits(value), value)
            for code in range(256):
                if code & 127 != 127:
                    self.assertEqual(native.decode_host(code), ref.e4m3_decode(code))

    def test_all_finite_codes_roundtrip(self):
        for code in range(256):
            if code & 127 == 127:
                continue
            self.assertEqual(ref.e4m3_encode(ref.e4m3_decode(code)), code)

    def test_zero_and_negative_zero(self):
        self.assertEqual(ref.e4m3_encode(0.0), 0)
        self.assertEqual(ref.e4m3_encode(-0.0), 128)
        self.assertEqual(math.copysign(1, ref.e4m3_decode(128)), -1)
        self.assertEqual(ref.bf16_bits(-0.0), 0x8000)

    def test_every_midpoint_ties_to_even_and_neighbors(self):
        for low in range(126):
            midpoint = (ref.e4m3_decode(low) + ref.e4m3_decode(low + 1)) / 2
            even = low if low % 2 == 0 else low + 1
            self.assertEqual(ref.e4m3_encode(midpoint), even)
            self.assertEqual(ref.e4m3_encode(-midpoint), even | 128)
            self.assertEqual(ref.e4m3_encode(adjacent_fp32(midpoint, -1)), low)
            self.assertEqual(ref.e4m3_encode(adjacent_fp32(midpoint, 1)), low + 1)

    def test_subnormal_boundary_max_and_saturation(self):
        self.assertEqual(ref.e4m3_decode(1), 2 ** -9)
        self.assertEqual(ref.e4m3_decode(8), 2 ** -6)
        self.assertEqual(ref.e4m3_decode(126), 448)
        self.assertEqual(ref.e4m3_encode(1e20), 126)
        self.assertEqual(ref.e4m3_encode(-1e20), 254)

    def test_nonfinite_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                ref.e4m3_encode(value)
        self.assertTrue(math.isnan(ref.e4m3_decode(127)))

    def test_bf16_known_ties(self):
        self.assertEqual(ref.bf16_round(1 + 2 ** -8), 1)
        self.assertEqual(ref.bf16_round(1 + 3 * 2 ** -8), 1 + 2 ** -6)
        self.assertEqual(ref.bf16_round(-(1 + 2 ** -8)), -1)


class TacoOracleTest(unittest.TestCase):
    def test_dense_transform_orthogonal(self):
        randomizer = random.Random(7321)
        values = [randomizer.uniform(-1, 1) for _ in range(128)]
        reconstructed = ref.hadamard_fp64(ref.hadamard_fp64(values))
        self.assertLess(max(abs(a - b) for a, b in zip(values, reconstructed)), 1e-12)
        self.assertAlmostEqual(math.fsum(x*x for x in values),
                               math.fsum(x*x for x in ref.hadamard_fp64(values)), places=11)

    def test_constant_and_impulse(self):
        constant = ref.hadamard_fp64([1.0] * 128)
        self.assertAlmostEqual(constant[0], math.sqrt(128))
        self.assertTrue(all(value == 0 for value in constant[1:]))
        impulse = ref.hadamard_fp64([1.0] + [0.0] * 127)
        self.assertTrue(all(abs(value - 1/math.sqrt(128)) < 1e-15 for value in impulse))

    def test_zero_packet_and_exact_layout(self):
        packet = ref.encode([0.0] * 256)
        self.assertEqual(len(packet), 272)
        self.assertEqual(packet[:256], bytes(256))
        self.assertEqual(struct.unpack_from("<ff", packet, 256), (ref.fp32(1e-12),) * 2)
        self.assertEqual(struct.unpack_from("<ff", packet, 264), (1000.0,) * 2)
        self.assertEqual(ref.decode(packet), [0.0] * 256)

    def test_groupwise_layout_not_interleaved_packets(self):
        first = [1.0] * 128
        second = [(-1.0 if index % 2 else 1.0) * 9 for index in range(128)]
        q0, s0, a0 = ref.encode_group(first)
        q1, s1, a1 = ref.encode_group(second)
        packet = ref.encode(first + second)
        self.assertEqual(packet, q0 + q1 + struct.pack("<ffff", s0, s1, a0, a1))

    def test_random_and_outlier_roundtrip(self):
        randomizer = random.Random(99)
        cases = [[randomizer.gauss(0, 1) for _ in range(128)],
                 [1000.0] + [randomizer.uniform(-0.01, 0.01) for _ in range(127)],
                 [1e-8 * (-1 if index % 2 else 1) for index in range(128)]]
        for values in cases:
            expected = [ref.bf16_round(value) for value in values]
            self.assertLess(ref.relative_l2(ref.decode(ref.encode(values)), expected), 0.05)

    def test_invalid_shape_and_corrupt_scales(self):
        for values in ([], [0.0]*127, [0.0]*129, [math.inf]*128):
            with self.assertRaises(ValueError):
                ref.encode(values)
        original = bytearray(ref.encode([1.0] * 128))
        for offset, value in ((128, math.nan), (128, -1.0), (132, 0.0)):
            packet = original.copy()
            struct.pack_into("<f", packet, offset, value)
            with self.assertRaises(ValueError):
                ref.decode(packet)
        packet = original.copy()
        packet[0] = 127
        with self.assertRaises(ValueError):
            ref.decode(packet)

    def test_ordered_reduction_source_placement(self):
        # BF16 sequential source sum differs from adding in arbitrary arrival order.
        sources = [[256.0], [1.0], [-256.0], [1.0]]
        for rank in range(4):
            remote = sources[:rank] + sources[rank + 1:]
            self.assertEqual(ref.reduce_sources(sources[rank], remote, rank, 4), [1.0])
        reordered = [sources[0], sources[2], sources[1], sources[3]]
        self.assertEqual(ref.reduce_sources(reordered[0], reordered[1:], 0, 4), [2.0])

    def test_packet_bytes_contract(self):
        self.assertEqual(packet_bytes(128), 136)
        self.assertEqual(packet_bytes(256 * 4096), 1114112)
        for elements in (0, -128, 129, 128.0, True):
            with self.assertRaises(ValueError):
                packet_bytes(elements)

    def test_import_does_not_import_torch(self):
        script = ("import sys; sys.path.insert(0, " + repr(str(ROOT / 'python')) + "); "
                  "import flux_dcu.reference, flux_dcu.codec; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
