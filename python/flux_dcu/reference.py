"""Standard-library CPU oracle for TACO H128/E4M3FN and ordered BF16 sums.

The Hadamard transform is an explicit FP64 Walsh matrix dot product, independent
of the HIP butterfly implementation. FP8 encoding enumerates all finite positive
codes and resolves exact ties by the low code bit. This module imports no torch.
Use small sampled groups for GPU validation: the independent transform is O(128²).
"""
from __future__ import annotations

import math
import struct

GROUP_SIZE = 128
GROUP_BYTES = 136


def fp32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", fp32(value)))[0]
    if (bits & 0x7fffffff) > 0x7f800000:
        return (bits >> 16) | 0x0040
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def bf16_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<I", bf16_bits(value) << 16))[0]


def e4m3_decode(code: int) -> float:
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("An E4M3FN code must be an integer byte")
    magnitude = code & 127
    if magnitude == 127:
        result = math.nan
    elif magnitude < 8:
        result = magnitude * 2.0 ** -9
    else:
        result = (1 + (magnitude & 7) / 8) * 2.0 ** ((magnitude >> 3) - 7)
    return math.copysign(result, -1 if code & 128 else 1)


_FP8_VALUES = tuple(e4m3_decode(code) for code in range(127))
_HADAMARD_SIGNS = tuple(tuple(-1.0 if (row & col).bit_count() & 1 else 1.0
                             for col in range(GROUP_SIZE)) for row in range(GROUP_SIZE))


def e4m3_encode(value: float) -> int:
    """FP32 input, finite saturation, RNE ties-to-even; signed zero preserved."""
    value = fp32(value)
    if not math.isfinite(value):
        raise ValueError("FP8 oracle accepts finite values only")
    sign = 128 if math.copysign(1.0, value) < 0 else 0
    magnitude = min(abs(value), 448.0)
    closest = min(range(127), key=lambda code: (abs(_FP8_VALUES[code] - magnitude), code & 1))
    return sign | closest


def hadamard_fp64(values) -> list[float]:
    values = [float(value) for value in values]
    if len(values) != GROUP_SIZE or not all(math.isfinite(x) for x in values):
        raise ValueError("Hadamard oracle requires exactly 128 finite values")
    inverse_norm = 1.0 / math.sqrt(GROUP_SIZE)
    return [math.fsum(sign * x for sign, x in zip(row, values)) * inverse_norm
            for row in _HADAMARD_SIGNS]


def encode_group(values):
    """Return (128 payload bytes, FP32 quant scale, FP32 adaptive scale).

    Inputs represent BF16 storage and are rounded accordingly. Matrix arithmetic
    intentionally uses FP64; packet bytes near rounding thresholds need not be
    bit-identical to the device's FP32 reduction order.
    """
    values = [bf16_round(value) for value in values]
    if len(values) != GROUP_SIZE or not all(math.isfinite(x) for x in values):
        raise ValueError("Encoding requires exactly 128 finite BF16-representable values")
    variance = max(math.fsum(x * x for x in values) / GROUP_SIZE + 1e-6, 1e-12)
    adaptive = fp32(min(max(448.0 / math.sqrt(variance), 1e-3), 1e3))
    transformed = hadamard_fp64([x * adaptive for x in values])
    quant = fp32(min(max(max(max(abs(x) for x in transformed), 1e-12) / 448.0, 1e-12), 1e6))
    payload = bytes(e4m3_encode(min(max(value / quant, -448.0), 448.0)) for value in transformed)
    return payload, quant, adaptive


def decode_group(payload, quant: float, adaptive: float) -> list[float]:
    payload = bytes(payload)
    if len(payload) != GROUP_SIZE:
        raise ValueError("A TACO group must contain exactly 128 coefficient bytes")
    quant, adaptive = float(quant), float(adaptive)
    if not (math.isfinite(quant) and fp32(1e-12) <= quant <= 1e6
            and math.isfinite(adaptive) and fp32(1e-3) <= adaptive <= 1e3):
        raise ValueError("Corrupt TACO scales")
    coefficients = [e4m3_decode(value) for value in payload]
    if not all(math.isfinite(x) for x in coefficients):
        raise ValueError("NaN E4M3FN payload is invalid in the finite TACO protocol")
    reconstructed = hadamard_fp64([value * quant for value in coefficients])
    return [bf16_round(value / adaptive) for value in reconstructed]


def encode(values) -> bytes:
    values = list(values)
    if not values or len(values) % GROUP_SIZE:
        raise ValueError("Encoding requires a positive multiple of 128 values")
    groups = [encode_group(values[start:start + GROUP_SIZE])
              for start in range(0, len(values), GROUP_SIZE)]
    return (b"".join(group[0] for group in groups)
            + struct.pack("<" + "f" * len(groups), *(group[1] for group in groups))
            + struct.pack("<" + "f" * len(groups), *(group[2] for group in groups)))


def decode(packet) -> list[float]:
    packet = bytes(packet)
    if not packet or len(packet) % GROUP_BYTES:
        raise ValueError("Packet must contain exactly 136 bytes per H128 group")
    groups = len(packet) // GROUP_BYTES
    elements = groups * GROUP_SIZE
    quant = struct.unpack_from("<" + "f" * groups, packet, elements)
    adaptive = struct.unpack_from("<" + "f" * groups, packet, elements + 4 * groups)
    result = []
    for group in range(groups):
        result.extend(decode_group(packet[group * GROUP_SIZE:(group + 1) * GROUP_SIZE],
                                   quant[group], adaptive[group]))
    return result


def reduce_sources(local, remote, rank: int, world: int) -> list[float]:
    """Source order 0..world-1, remote sources ascending excluding rank.

    Every input and each intermediate addition is rounded to BF16. local and
    each remote source are flat sequences representing equally sized matrices.
    """
    if type(rank) is not int or type(world) is not int or not 2 <= world <= 128 or not 0 <= rank < world:
        raise ValueError("Invalid rank/world")
    local = [bf16_round(value) for value in local]
    remote = [[bf16_round(value) for value in row] for row in remote]
    if not local or len(remote) != world - 1 or any(len(row) != len(local) for row in remote):
        raise ValueError("Expected world-1 remote sources with the same nonzero element count")
    if not all(math.isfinite(value) for row in [local, *remote] for value in row):
        raise ValueError("Reduction oracle requires finite inputs")
    sources = remote[:rank] + [local] + remote[rank:]
    output = [0.0] * len(local)
    for source in sources:
        output = [bf16_round(fp32(accum + value)) for accum, value in zip(output, source)]
    return output


def relative_l2(actual, expected) -> float:
    actual, expected = list(actual), list(expected)
    if len(actual) != len(expected) or not actual:
        raise ValueError("Expected equal nonempty vectors")
    if not all(math.isfinite(float(x)) for x in actual + expected):
        return math.inf
    numerator = math.fsum((a - b) ** 2 for a, b in zip(actual, expected))
    denominator = math.fsum(b * b for b in expected)
    return math.sqrt(numerator / denominator) if denominator else (0.0 if numerator == 0 else math.inf)
