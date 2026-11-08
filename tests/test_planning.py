import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from flux_dcu import planning as p


class PlanContract(unittest.TestCase):
    shape = (8, 128, 16, 2, 2)

    def test_orders_are_legal_rank_independent_and_deduplicated(self):
        for world in (2, 4, 8):
            shape = (4 * world, 128, 16, world, 2)
            candidates = p.candidates(*shape)
            self.assertEqual(candidates[0]['arrival_order'], p.base_order(*shape))
            self.assertEqual(len({tuple(c['arrival_order']) for c in candidates}), len(candidates))
            for candidate in candidates:
                plan = p.make_role(*shape, order=candidate['arrival_order'])
                self.assertIs(p.validate_role_plan(plan, *shape), plan)
                self.assertEqual([block // 2 for block in plan['arrival_order']],
                                 [slot % world for slot in range(2 * world)])

    def test_runtime_keyword_signature_and_invalid_plans(self):
        plan = p.make_role(*self.shape)
        self.assertIs(p.validate_role_plan(plan, shape=[8, 128, 16], world_size=2, chunk_rows=2), plan)
        for key, value in [('shape', [8, 128, 17]), ('chunk_rows', 4),
                           ('arrival_order', [0, 1, 2, 3]),
                           ('arrival_order', [0, 2, 0, 3]),
                           ('selective_mask', [1, 0, 1, 0])]:
            changed = dict(plan, **{key: value})
            with self.assertRaises(ValueError):
                p.validate_role_plan(changed, *self.shape)

    def test_closure_and_exact_source_byte_accounting(self):
        plan = p.make_role(*self.shape, mask=[True, False, False, False])
        stats = p.quantization_statistics(plan, 2)
        self.assertEqual(p.quantized_blocks(plan), [0])
        self.assertEqual(stats['actual_remote_fraction'], .25)
        self.assertEqual(stats['quantized_remote_contributions'], 1)
        self.assertEqual(stats['selected_per_source'], [0, 1])
        self.assertEqual(stats['remote_wire_bytes_per_source'], [1024, 784])
        self.assertEqual(stats['mixed_source_blocks'], 0)

    def test_default_h128_counts(self):
        plan = p.make_role(2048, 4096, 512, 8, 128)
        stats = p.quantization_statistics(plan, 8)
        self.assertEqual(stats['remote_wire_bytes_per_source'], [7798784] * 8)
        plan['selective_mask'] = [False] * 16
        stats = p.quantization_statistics(plan, 8)
        self.assertEqual(stats['remote_wire_bytes_per_source'], [14680064] * 8)

    def test_digest_has_stable_json_and_catches_changes(self):
        plan = p.make_role(*self.shape)
        self.assertEqual(p.role_digest(plan), p.role_digest(json.loads(json.dumps(plan))))
        altered = dict(plan, selective_mask=[False] * 4)
        self.assertNotEqual(p.role_digest(plan), p.role_digest(altered))
        altered = dict(plan, diagnostics={'time': float('nan')})
        with self.assertRaises(ValueError):
            p.role_digest(altered)


class MeasuredSelection(unittest.TestCase):
    shape = (8, 128, 16, 2, 2)
    cfg = dict(warmup=0, iterations=1, training_pairs=3, confirmation_pairs=3,
               min_gain=.002, quant_budget=.5)

    def evaluator(self, *, reject_arrival=False, reject_mask=False, noise=False):
        calls = []
        reverse = p.candidates(*self.shape)[1]['arrival_order']

        def measure(plan, seed, arm, label):
            calls.append(dict(plan=json.loads(json.dumps(plan)), seed=seed, arm=arm, label=label))
            if arm == p.ARRIVAL_ARM:
                latency = 90. if plan['arrival_order'] == reverse else 100.
                if reject_arrival and label.startswith('arrival_confirmation') and '/candidate' in label:
                    latency = 110.
            else:
                latency = 100. - 3. * sum(plan['selective_mask'])
                if reject_mask and label.startswith('selective_confirmation') and '/candidate' in label:
                    latency = 101.
            if noise and 'confirmation' in label and '/pair1/candidate' in label:
                latency = 101.
            return latency
        return measure, calls

    def test_real_measurement_selects_order_and_budgeted_mask(self):
        evaluate, calls = self.evaluator()
        plan = p.fit_role(*self.shape, evaluate, config=self.cfg)
        self.assertTrue(plan['diagnostics']['arrival']['accepted'])
        self.assertTrue(plan['diagnostics']['selective']['accepted'])
        self.assertEqual(sum(plan['selective_mask']), 2)
        self.assertEqual(plan['selective_mask'], [True, True, False, False])
        self.assertLessEqual(sum(plan['selective_mask']), 4 * self.cfg['quant_budget'])
        # The full control wins in raw latency but cannot exceed a .5 budget.
        self.assertGreater(plan['diagnostics']['selective']['full_control']['fractional_gain'],
                           plan['diagnostics']['selective']['confirmation']['fractional_gain'])
        for record in plan['diagnostics']['arrival']['training_candidates']:
            self.assertEqual(record['paired_samples'][0]['execution_order'], ['reference', 'candidate'])
            self.assertEqual(record['paired_samples'][1]['execution_order'], ['candidate', 'reference'])

    def test_confirmation_does_not_reselect_another_candidate(self):
        evaluate, calls = self.evaluator(reject_arrival=True, reject_mask=True)
        plan = p.fit_role(*self.shape, evaluate, config=self.cfg)
        self.assertEqual(plan['arrival_order'], p.base_order(*self.shape))
        self.assertFalse(any(plan['selective_mask']))
        self.assertTrue(plan['diagnostics']['arrival']['identity_fallback'])
        self.assertTrue(plan['diagnostics']['selective']['zero_fallback'])
        for phase in ('arrival_confirmation', 'selective_confirmation'):
            candidates = [c for c in calls if c['label'].startswith(phase) and '/candidate' in c['label']]
            self.assertEqual(len({p.role_digest(c['plan']) for c in candidates}), 1)
        fit_seeds = {c['seed'] for c in calls if 'confirmation' not in c['label']}
        heldout_seeds = {c['seed'] for c in calls if 'confirmation' in c['label']}
        self.assertFalse(fit_seeds & heldout_seeds)

    def test_unstable_confirmation_pair_rejects_optimistic_average(self):
        evaluate, _ = self.evaluator(noise=True)
        plan = p.fit_role(*self.shape, evaluate, config=self.cfg)
        self.assertFalse(plan['diagnostics']['arrival']['accepted'])
        self.assertFalse(plan['diagnostics']['selective']['accepted'])

    def test_zero_budget_still_measures_full_control_without_using_it(self):
        evaluate, calls = self.evaluator()
        plan = p.fit_role(*self.shape, evaluate, config=dict(self.cfg, quant_budget=0))
        self.assertFalse(any(plan['selective_mask']))
        self.assertFalse(plan['diagnostics']['selective']['singletons'])
        self.assertTrue(any(c['label'].startswith('selective_full_control') for c in calls))
        self.assertFalse(any(c['label'].startswith('selective_confirmation') for c in calls))

    def test_full_budget_can_accept_full_quantization(self):
        evaluate, _ = self.evaluator()
        plan = p.fit_role(*self.shape, evaluate, config=dict(self.cfg, quant_budget=1))
        self.assertTrue(all(plan['selective_mask']))
        self.assertEqual(plan['diagnostics']['actual_selection']['actual_remote_fraction'], 1.)

    def test_invalid_measurements_and_numerical_control_failure_are_fatal(self):
        for value in (0., -1., float('nan'), float('inf'), {'elapsed_seconds': 1., 'correctness_passed': False}):
            with self.assertRaises(ValueError):
                p.fit_role(*self.shape, lambda *args: value, config=self.cfg)
        evaluate, _ = self.evaluator()

        def bad_control(plan, seed, arm, label):
            if label.startswith('selective_full_control'):
                raise RuntimeError('full quantization failed numerical tolerance')
            return evaluate(plan, seed, arm, label)
        with self.assertRaisesRegex(RuntimeError, 'numerical tolerance'):
            p.fit_role(*self.shape, bad_control, config=dict(self.cfg, quant_budget=0))

    def test_settings_validation(self):
        for kwargs in ({'quant_budget': -1}, {'quant_budget': 1.1}, {'min_gain': 1},
                       {'training_pairs': 0}, {'confirmation_pairs': True}, {'random_field': 1}):
            with self.assertRaises(ValueError):
                p.settings(kwargs)


if __name__ == '__main__':
    unittest.main()
