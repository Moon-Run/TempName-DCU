"""The GPU preflight oracle is also checked against an independent CPU oracle."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU PyTorch not installed')
class FrameworkReferenceTests(unittest.TestCase):
    def test_selective_reference_uses_physical_closed_blocks(self):
        import torch
        from flux_dcu.worker import reduce_reference_sources, independent_codec
        generator = torch.Generator().manual_seed(43)
        sources = torch.randn((2, 4, 128), generator=generator).to(torch.bfloat16)
        plain, zero = reduce_reference_sources(sources, 0, True, [False] * 4, 2)
        self.assertTrue(torch.equal(plain, zero))
        _, selected = reduce_reference_sources(sources, 0, True, [True, False, False, False], 2)
        expected_remote = sources[1].clone()
        expected_remote[:2] = independent_codec(expected_remote[:2])
        self.assertTrue(torch.equal(selected, sources[0] + expected_remote))
        self.assertTrue(torch.equal(selected[2:], plain[2:]))
        _, irrelevant = reduce_reference_sources(sources, 0, True, [False, False, True, True], 2)
        self.assertTrue(torch.equal(plain, irrelevant))
        _, full = reduce_reference_sources(sources, 0, True, [True] * 4, 2)
        _, original_full = reduce_reference_sources(sources, 0, True)
        self.assertTrue(torch.equal(full, original_full))

    def test_model_receives_role_specific_frozen_plans(self):
        from unittest.mock import patch
        import torch
        import torch.distributed as dist
        from flux_dcu.model import build_model
        from flux_dcu.planning import make_role
        config = dict(layers=2, hidden=256, ffn=1024, heads=8, sequence=128,
                      micro_batch=1, accumulation_steps=1, vocab=1024, chunk_rows=32)
        bundle = dict(roles={role: make_role(128, 256, k, 2, 32)
                             for role, k in [('attn', 128), ('mlp', 512)]})
        received = []
        class Recorder:
            def __init__(self, **kwargs):
                received.append(kwargs)
        with patch.object(dist, 'is_initialized', return_value=True), \
             patch.object(dist, 'get_world_size', return_value=2), \
             patch.object(dist, 'get_rank', return_value=0):
            model = build_model(config, 'flux_taco_interleaved_arrival_selective',
                                device='cpu', operator_factory=Recorder, execution_plan=bundle)
        self.assertEqual(len(model.operators), 4)
        for index, kwargs in enumerate(received):
            self.assertEqual(kwargs['plan'], bundle['roles']['attn' if index % 2 == 0 else 'mlp'])

    def test_high_level_codec_matches_fp64_matrix_oracle(self):
        import torch
        from flux_dcu.reference import encode, decode
        from flux_dcu.worker import independent_codec, relative_error
        rng = random.Random(1197)
        cases = [[0.] * 256, [rng.gauss(0, 1) for _ in range(256)],
                 [64. if i == 0 else (-96. if i == 128 else .01) for i in range(256)]]
        for values in cases:
            data = torch.tensor(values, dtype=torch.bfloat16).reshape(2, 128)
            expected = torch.tensor(decode(encode(data.float().flatten().tolist()))).reshape(2, 128)
            actual = independent_codec(data)
            self.assertLessEqual(relative_error(actual, expected), .02)

    def test_tensor_hash_is_bit_sensitive(self):
        import torch
        from flux_dcu.worker import tensor_sha
        a = torch.zeros(8, dtype=torch.bfloat16)
        b = a.clone(); b[0] = 1
        self.assertEqual(tensor_sha(a), tensor_sha(a.clone()))
        self.assertNotEqual(tensor_sha(a), tensor_sha(b))


if __name__ == '__main__':
    unittest.main()
