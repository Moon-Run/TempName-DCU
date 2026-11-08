"""CPU regression checks for experiment attribution and failure handling."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1] / 'tools/dcu/phase6'
sys.path.insert(0, str(HERE))
import protocol as p
import report as reporting

spec = importlib.util.spec_from_file_location('dcu_controller_under_test', HERE / 'init.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ProtocolTests(unittest.TestCase):
    def test_ten_blocks_balance_position_and_pair_order(self):
        blocks = p.orders(1, 10)
        for arm in p.ARMS:
            self.assertEqual([sum(row['arms'][i] == arm for row in blocks) for i in range(5)], [2] * 5)
        for a in p.ARMS:
            for b in p.ARMS:
                if a != b:
                    self.assertEqual(sum(row['arms'].index(a) < row['arms'].index(b) for row in blocks), 5)

    def test_short_run_has_no_confidence_interval(self):
        value = reporting.pair_statistics([9.0], [10.0], balanced=False)
        self.assertIsNone(value['ci95'])
        self.assertAlmostEqual(value['percent'], 10)
        self.assertFalse(p.balanced(7))

    def test_paired_statistics_preserve_regression(self):
        value = reporting.pair_statistics([11.] * 10, [10.] * 10, balanced=True, bootstrap=100)
        self.assertAlmostEqual(value['percent'], -10)
        self.assertTrue(all(v < 0 for v in value['ci95']))
        self.assertEqual(value['faster_blocks'], 0)

    def test_scope_and_work_counts(self):
        self.assertEqual(len(p.jobs(1, 1, 'preflight')), 1)
        self.assertEqual(len(p.jobs(1, 1, 'operator')), 7)
        jobs = p.jobs(2, 10, 'train')
        self.assertEqual(sum(j['stage'] == 'train' for j in jobs), 100)
        self.assertEqual(sum(j['stage'] == 'operator' for j in jobs), 100)
        self.assertEqual(sum(j['stage'] == 'calibrate' for j in jobs), 1)
        self.assertEqual(len({j['directory'] for j in jobs}), len(jobs))

    def test_invalid_dimensions_and_identity_schedule_rejected(self):
        with self.assertRaises(ValueError):
            controller.arguments(['--chunk-rows', '256'])
        with self.assertRaises(ValueError):
            controller.arguments(['--chunk-rows', '31'])
        with self.assertRaises(ValueError):
            controller.arguments(['--blocks', '0'])
        with self.assertRaises(ValueError):
            controller.arguments(['--warmup-steps', '0'])
        with self.assertRaises(ValueError):
            controller.arguments(['--quant-budget', '1.1'])
        with self.assertRaises(ValueError):
            controller.arguments(['--calibration-pairs', '0'])
        self.assertEqual(controller.arguments(['--operator-only', '--warmup-steps', '0']).scope, 'operator')
        for world in (2, 4, 8):
            args = controller.arguments(['--small-model', '--world-size', str(world)])
            self.assertEqual(args.model['sequence'] // world // args.model['chunk_rows'], 2)

    def test_preview_imports_no_torch_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'never-created'
            code = ('import sys,runpy; from pathlib import Path; '
                    'script=sys.argv.pop(1);sys.argv[0]=script;'
                    'sys.path.insert(0,str(Path(script).parent));'
                    'runpy.run_path(script,run_name="__main__");'
                    'assert "torch" not in sys.modules, "preview imported torch"')
            result = subprocess.run([sys.executable, '-B', '-c', code, str(HERE / 'init.py'),
                '--trials', '1', '--blocks', '1', '--out', str(output)],
                capture_output=True, text=True, check=True)
            value = json.loads(result.stdout)
            self.assertTrue(value['preview'])
            self.assertFalse(value['uses_gpu'])
            self.assertFalse(output.exists())

    def test_sh_entry_is_preview_only(self):
        result = subprocess.run(['bash', str(HERE / 'run_init.sh'), '--blocks', '1',
                                 '--small-model'], capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value['scope'], 'train')
        self.assertEqual(value['config']['model']['layers'], 2)
        self.assertTrue(value['preview'])
        self.assertEqual(len(value['jobs']), 12)
        self.assertIsNone(value['config']['execution_plan'])

    def test_snapshot_includes_both_upstream_reference_trees(self):
        files = {str(file.relative_to(controller.REPO)) for file in controller.source_files()}
        self.assertIn('third_party/flux/provenance.json', files)
        self.assertIn('third_party/flux/src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp', files)
        self.assertIn('third_party/taco/upstream/taco.cu', files)
        self.assertEqual(set(reporting.validate_reference_sources(controller.REPO)), {'flux', 'taco'})

    def test_trajectory_mismatch_is_not_accepted(self):
        a = [dict(initialization_sha256='a' * 64, input_sha256='b' * 64,
                  losses=[1.0], grad_norms=[2.0])]
        b = [dict(a[0], losses=[1.0001])]
        with self.assertRaisesRegex(ValueError, 'losses'):
            p.compare_taco_trajectories(a, b)

    def test_timeout_cleanup_targets_only_owned_process_group(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('fixture', 10), 0]
        with mock.patch.object(controller.os, 'killpg') as kill:
            controller.stop_process(process)
        self.assertEqual(kill.call_args_list, [mock.call(12345, controller.signal.SIGTERM),
                                               mock.call(12345, controller.signal.SIGKILL)])


class ReportTests(unittest.TestCase):
    def fixture(self, root, profile=False):
        arguments = ['--blocks', '1', '--steps', '2', '--operator-iters', '3',
                     '--world-size', '2', '--small-model', '--out', str(root)]
        if profile:
            arguments.append('--profile')
        args = controller.arguments(arguments)
        source = root / 'sources/src/taco/codec.hip'
        source.parent.mkdir(parents=True)
        source.write_text('# Test fixture only\n')
        manifest = {'src/taco/codec.hip': p.sha(source)}
        planning_source = root / 'sources/python/flux_dcu/planning.py'
        planning_source.parent.mkdir(parents=True)
        planning_source.write_bytes((controller.REPO / 'python/flux_dcu/planning.py').read_bytes())
        manifest['python/flux_dcu/planning.py'] = p.sha(planning_source)
        for name, filename, revision_key, revision in (
            ('flux', 'provenance.json', 'commit', '19831ca2d820e3e782ed1d15d8b52d0898b78b26'),
            ('taco', 'manifest.json', 'source_revision', 'ca717e05b6dd6d21fff4d2617d209c95be63f3ef')):
            directory = root / 'sources/third_party' / name
            directory.mkdir(parents=True)
            reference = directory / 'fixture.txt'
            reference.write_text('Synthetic provenance fixture, never a production archive.')
            provenance = directory / filename
            p.write(provenance, {revision_key: revision, 'files': {'fixture.txt': p.sha(reference)}})
            for file in (reference, provenance):
                manifest[str(file.relative_to(root / 'sources'))] = p.sha(file)
        plan = controller.plan_for(args, manifest, run_id='fixture-run')
        p.write(root / 'source-manifest.json', manifest)
        p.write(root / 'plan.json', plan)
        build_identity = dict(source_sha256=p.sha(source), abi='fixture')
        build_key = hashlib.sha256(json.dumps(build_identity, sort_keys=True).encode()).hexdigest()[:24]
        library = root / 'build' / build_key / 'libflux_dcu_taco.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'NOT-A-LIBRARY: UNIT TEST FIXTURE')
        codec_build = dict(identity=build_identity, library_sha256=p.sha(library))
        p.write(library.parent / 'manifest.json', codec_build)
        model, timing = plan['config']['model'], plan['config']['timing']
        planning_spec = importlib.util.spec_from_file_location('dcu_fixture_planner', planning_source)
        planner = importlib.util.module_from_spec(planning_spec)
        planning_spec.loader.exec_module(planner)
        calibration_samples, roles = [], {}
        for role_index, role in enumerate(('attn', 'mlp')):
            shape = [model['sequence'] * model['micro_batch'], model['hidden'],
                     model['hidden' if role == 'attn' else 'ffn'] // 2]
            baseline_order = planner.base_order(*shape, 2, model['chunk_rows'])
            def evaluate(candidate, seed, arm, label):
                # Deterministic injected timing is a unit-test oracle, never a
                # real device measurement or a document-derived success flag.
                elapsed = (.01 if candidate['arrival_order'] == baseline_order else .008) if arm == p.ARRIVAL else (
                    .01 - .0005 * sum(candidate['selective_mask']))
                calibration_samples.append(dict(role=role, label=label, seed=seed, arm=arm,
                    plan_digest=planner.role_digest(candidate), arrival_order=candidate['arrival_order'],
                    selective_mask=candidate['selective_mask'], local_elapsed_seconds=elapsed,
                    max_rank_elapsed_seconds=elapsed, iterations=plan['config']['calibration']['iterations'],
                    warmup=plan['config']['calibration']['warmup'], bf16_relative_l2=.001,
                    codec_reference_relative_l2=.001, correctness_passed=True))
                return elapsed
            roles[role] = planner.fit_role(*shape, 2, model['chunk_rows'], evaluate,
                config=plan['config']['calibration'], seed=8000000 + 10000000 * role_index + plan['config']['seed'])
        bundle = dict(schema=p.PLAN_SCHEMA, world_size=2, transport_lanes=2, roles=roles)

        def state(arm, rank, role, calls, stage):
            role_plan = roles[role] if arm in (p.ARRIVAL, p.SELECTIVE) else None
            schedule = p.expected_schedule(model, 2, arm, role_plan)
            mask = roles[role]['selective_mask'] if arm == p.SELECTIVE else [arm != 'flux'] * len(schedule)
            per = len(schedule) // 2
            selected = [sum(flag and physical // per != source for physical, flag in enumerate(mask)) for source in range(2)]
            elements = model['chunk_rows'] * model['hidden']
            wires = [q * (elements // 128 * 136) + (per - q) * elements * 2 for q in selected]
            return dict(backend=p.BACKEND, arm=arm, rank=rank, calls=calls,
                shape=[model['sequence'] * model['micro_batch'], model['hidden'],
                       model['hidden' if role == 'attn' else 'ffn'] // 2], chunk_rows=model['chunk_rows'],
                production_schedule=schedule,
                mapping_sha256=hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
                gemms_per_call=len(schedule), collectives_per_call=len(schedule),
                quantization=('selective_remote_h128_e4m3' if arm == p.SELECTIVE else
                              'none' if arm == 'flux' else 'all_remote_h128_e4m3'),
                reduction_order=[0, 1], rounding='bf16_after_each_source',
                selection_mask=mask, selected_remote_blocks_by_source=selected,
                remote_wire_bytes_by_source=wires, remote_wire_bytes_per_source=wires[rank],
                actual_remote_wire_bytes_per_rank=wires[rank], plan_sha256=p.digest(role_plan) if role_plan else None,
                scratch_bytes=1024, original_cuda_flux=False, cta_level_swizzle=False,
                shared_streams_per_group=5, transport_lanes=2,
                consumer_policy='per_lane_ready' if arm in (p.ARRIVAL, p.SELECTIVE) else 'ordered',
                diagnostic_delay_cycles=0, diagnostic_delay_in_normal_forward=False,
                validation_calls=2 if stage == 'operator' else 1)

        completed = []
        frozen_bundle = None
        for job in plan['jobs']:
            folder = root / 'results' / job['directory']
            job_config = p.config_for_job(plan['config'], job, frozen_bundle)
            p.write(folder / 'config.json', job_config)
            p.write(folder / 'command.json', dict(job=job, command=['fixture-only']))
            for rank in range(2):
                row = dict(rank=rank, world_size=2, stage=job['stage'], arm=job['arm'],
                    run_id='fixture-run', completed=True, passed=True,
                    environment_identity=dict(host='synthetic-fixture', device=rank),
                    backend=p.BACKEND, original_cuda_flux=False,
                    cta_level_swizzle=False, codec_build=codec_build,
                    execution_plan_sha256=job_config['execution_plan_sha256'])
                if job['stage'] == 'calibrate':
                    samples = [dict(sample, local_elapsed_seconds=sample['max_rank_elapsed_seconds'] * (.9 if rank == 0 else 1.))
                               for sample in calibration_samples]
                    raw_file = folder / f'calibration-rank{rank}-samples.json'
                    p.write(raw_file, dict(schema='dcu-calibration-samples-v2', rank=rank, samples=samples))
                    row.update(calibration_plan=bundle, calibration_results=dict(plan_sha256=p.digest(bundle),
                        raw_samples=samples, raw_samples_file=raw_file.name))
                elif job['stage'] == 'train':
                    row.update(elapsed_seconds=2.0 + rank, timed_steps=2, losses=[2., 1.],
                               grad_norms=[3., 2.], initialization_sha256=str(rank) * 64,
                               input_sha256='a' * 64, final_parameters_sha256='b' * 64,
                               actual_model_config=model, parameter_counts=p.expected_parameter_counts(model, 2),
                               actual_parameters_per_rank=p.expected_parameter_counts(model, 2)['parameters_per_rank'],
                               warmup_steps=timing['warmup_steps'],
                               tokens_per_step=model['sequence'] * model['micro_batch'] * model['accumulation_steps'],
                               dtype='bfloat16', optimizer='AdamW-fp32-master',
                               operator_counters=dict(model_forwards=2, calls_per_projection=2,
                                   attention_projection_calls=2 * model['layers'], mlp_projection_calls=2 * model['layers'],
                                   target_modules=2 * model['layers'], fallback_calls=0,
                                   actual_calls_per_instance=[2] * (2 * model['layers']),
                                   actual_total_projection_calls=4 * model['layers']),
                               operator_states=[state(job['arm'], rank, role, 2 + timing['warmup_steps'], 'train')
                                                for _ in range(model['layers']) for role in ('attn', 'mlp')],
                               memory=dict(peak_allocated_bytes=1024, peak_reserved_bytes=2048))
                    if job['arm'] == p.SELECTIVE:
                        row.update(losses=[1.9, .95], final_parameters_sha256='e' * 64)
                elif job['stage'] == 'operator':
                    row.update(initialization_sha256=str(rank) * 64, input_sha256='a' * 64)
                    row['operator_results'] = []
                    for role in ('attn', 'mlp'):
                        diagnostic = None
                        if profile:
                            trace = folder / f'rank{rank}-{role}-trace.json'
                            trace.write_text('{"traceEvents": [], "synthetic_fixture": true}')
                            diagnostic = dict(trace=trace.name, trace_sha256=p.sha(trace),
                                              actual_kernel_overlap_validated=False)
                        row['operator_results'].append(dict(role=role, elapsed_seconds=.03 * (rank + 1),
                            timed_iterations=3, correctness_passed=True, bf16_relative_l2=.001,
                            codec_reference_relative_l2=.001, changed_input_relative_l2=.001,
                            full_gemm_only_seconds=.01, order_bitwise_equal=job['arm'] != 'flux',
                            profile=diagnostic, execution=state(job['arm'], rank, role,
                                2 + int(profile) + timing['operator_warmup'] + timing['operator_iterations'], 'operator')))
                        row['operator_results'][-1]['arrival_diagnostic'] = (dict(passed=True,
                            numerical_bitwise_equal=True, artificial=True, ordinary_speedup_claim=False,
                            delayed_rank=0, delayed_lane=0, delayed_physical=0, delay_cycles=50000000,
                            later_blocks_completed_before_pending=[], ready_bypass_observed=False,
                            evidence=dict(events=[dict(stage='communication', physical=0, lane=0,
                                                       start_ms=0., end_ms=1.)]))
                            if job['arm'] in (p.ARRIVAL, p.SELECTIVE) else None)
                p.write(folder / f'rank{rank}.json', row)
            done = dict(job=job, command_sha256=p.sha(folder / 'command.json'),
                rank_sha256={f'rank{rank}.json': p.sha(folder / f'rank{rank}.json') for rank in range(2)})
            if job['stage'] == 'calibrate':
                frozen_bundle = bundle
                p.write(root / 'selection-plan.json', bundle)
                done.update(execution_plan_sha256=p.digest(bundle), selection_plan_file_sha256=p.sha(root / 'selection-plan.json'))
            completed.append(done)
        p.write(root / 'completed-jobs.json', completed)
        return plan

    def test_max_rank_denominator_and_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = reporting.replay(root)
            self.assertEqual(result['formal_optimizer_steps'], 10)
            self.assertEqual(result['training_windows'], 5)
            self.assertFalse(result['ci_available'])
            self.assertTrue(result['full_quantized_trajectories_identical'])
            self.assertEqual(result['trajectory_equivalence_arms'], list(p.FULL_TACO_ARMS))
            self.assertEqual(result['samples']['train'][0]['value'], 1500.)
            self.assertEqual(result['samples']['operator'][0]['value'], 20000.)
            self.assertIn('短测', reporting.markdown(result))
            metrics = result['results']['train']['trial1']['full_step']['arms']['flux']
            self.assertEqual(metrics['mean_training_loss'], 1.5)
            self.assertEqual(metrics['mean_final_loss'], 1.)
            self.assertAlmostEqual(metrics['tokens_per_second'], 128 / 1.5)
            self.assertEqual(metrics['max_peak_allocated_gib'], 1024 / 2**30)
            self.assertEqual(result['selection']['roles']['attn']['actual_quantized_remote_fraction'], .25)
            self.assertGreater(result['selection']['roles']['attn']['remote_bytes_ratio_to_full_quantized'], 1)
            diagnostic = result['results']['operator']['trial1']['attn']['arms'][p.ARRIVAL]
            self.assertEqual(diagnostic['artificial_bypass_rank_windows'], 0)
            self.assertEqual(diagnostic['artificial_diagnostic_rank_windows'], 2)

    def rewrite_calibration(self, root, transform):
        completed = p.read(root / 'completed-jobs.json')
        item = next(entry for entry in completed if entry['job']['stage'] == 'calibrate')
        folder = root / 'results' / item['job']['directory']
        for rank in range(2):
            file = folder / f'rank{rank}.json'
            row = p.read(file)
            transform(row)
            p.write(folder / row['calibration_results']['raw_samples_file'],
                    dict(schema='dcu-calibration-samples-v2', rank=rank,
                         samples=row['calibration_results']['raw_samples']))
            p.write(file, row)
            item['rank_sha256'][file.name] = p.sha(file)
        p.write(root / 'completed-jobs.json', completed)

    def test_calibration_uses_real_max_rank_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            def alter(row):
                row['calibration_results']['raw_samples'][0]['max_rank_elapsed_seconds'] *= 2
            self.rewrite_calibration(root, alter)
            with self.assertRaisesRegex(ValueError, 'actual maximum rank'):
                reporting.replay(root)

    def test_confirmation_seed_drift_rejected_by_frozen_planner_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            def alter(row):
                sample = next(item for item in row['calibration_results']['raw_samples']
                              if item['label'].startswith('arrival_confirmation/'))
                sample['seed'] += 99
            self.rewrite_calibration(root, alter)
            with self.assertRaisesRegex(ValueError, 'candidate/seed/order'):
                reporting.replay(root)

    def test_calibration_diagnostics_cannot_replace_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            bundle = p.read(root / 'selection-plan.json')
            bundle['roles']['attn']['diagnostics']['unsupported_claim'] = 'A document says this is faster'
            def alter(row):
                row['calibration_plan'] = bundle
                row['calibration_results']['plan_sha256'] = p.digest(bundle)
            self.rewrite_calibration(root, alter)
            p.write(root / 'selection-plan.json', bundle)
            completed = p.read(root / 'completed-jobs.json')
            item = next(entry for entry in completed if entry['job']['stage'] == 'calibrate')
            item.update(execution_plan_sha256=p.digest(bundle),
                        selection_plan_file_sha256=p.sha(root / 'selection-plan.json'))
            p.write(root / 'completed-jobs.json', completed)
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                reporting.replay(root)

    def test_calibration_does_not_mutate_original_job_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.fixture(root)
            self.assertIsNone(plan['config']['execution_plan'])
            for directory in ('preflight', 'calibrate'):
                cfg = p.read(root / 'results' / directory / 'config.json')
                self.assertIsNone(cfg['execution_plan'])
                self.assertIsNone(cfg['execution_plan_sha256'])
            first_formal = next(job for job in plan['jobs'] if job['stage'] == 'operator')
            cfg = p.read(root / 'results' / first_formal['directory'] / 'config.json')
            self.assertEqual(cfg['execution_plan'], p.read(root / 'selection-plan.json'))
            self.assertEqual(cfg['execution_plan_sha256'], p.digest(cfg['execution_plan']))

    def test_arrival_training_must_match_full_quantized_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.fixture(root)
            completed = p.read(root / 'completed-jobs.json')
            index = next(i for i, job in enumerate(plan['jobs'])
                         if job['stage'] == 'train' and job['arm'] == p.ARRIVAL)
            file = root / 'results' / plan['jobs'][index]['directory'] / 'rank0.json'
            row = p.read(file)
            row['losses'][0] += .01
            p.write(file, row)
            completed[index]['rank_sha256'][file.name] = p.sha(file)
            p.write(root / 'completed-jobs.json', completed)
            with self.assertRaisesRegex(ValueError, 'changed losses'):
                reporting.replay(root)

    def test_missing_artificial_diagnostic_is_rejected_but_zero_bypass_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.fixture(root)
            completed = p.read(root / 'completed-jobs.json')
            index = next(i for i, job in enumerate(plan['jobs'])
                         if job['stage'] == 'operator' and job['arm'] == p.SELECTIVE)
            file = root / 'results' / plan['jobs'][index]['directory'] / 'rank0.json'
            row = p.read(file)
            row['operator_results'][0]['arrival_diagnostic'] = None
            p.write(file, row)
            completed[index]['rank_sha256'][file.name] = p.sha(file)
            p.write(root / 'completed-jobs.json', completed)
            with self.assertRaisesRegex(ValueError, 'independent artificial'):
                reporting.replay(root)

    def test_missing_rank_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'results/preflight/rank1.json').unlink()
            with self.assertRaises(FileNotFoundError):
                reporting.replay(root)

    def test_changed_record_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            with (root / 'results/preflight/rank0.json').open('a') as file:
                file.write(' ')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                reporting.replay(root)

    def test_completed_label_cannot_hide_missing_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            completed = p.read(root / 'completed-jobs.json')
            p.write(root / 'completed-jobs.json', completed[:-1])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                reporting.replay(root)

    def test_environment_drift_is_rejected_even_with_new_record_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.fixture(root)
            done = p.read(root / 'completed-jobs.json')
            folder = root / 'results' / plan['jobs'][1]['directory']
            row = p.read(folder / 'rank0.json')
            row['environment_identity']['host'] = 'different-host'
            p.write(folder / 'rank0.json', row)
            done[1]['rank_sha256']['rank0.json'] = p.sha(folder / 'rank0.json')
            p.write(root / 'completed-jobs.json', done)
            with self.assertRaisesRegex(ValueError, 'Hardware/software'):
                reporting.replay(root)

    def test_native_library_bytes_are_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            next((root / 'build').glob('*/libflux_dcu_taco.so')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'library bytes'):
                reporting.replay(root)

    def test_profile_files_are_hashed_not_promoted_to_overlap_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, profile=True)
            report = reporting.replay(root)
            self.assertFalse(report['actual_kernel_overlap_validated'])
            self.assertTrue(any('trace.json' in key for key in report['artifact_sha256']))
            next((root / 'results/operator').rglob('*trace.json')).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Profile trace bytes'):
                reporting.replay(root)

    def test_wrong_projection_count_rejected_even_if_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.fixture(root)
            completed = p.read(root / 'completed-jobs.json')
            index = next(i for i, j in enumerate(plan['jobs']) if j['stage'] == 'train')
            file = root / 'results' / plan['jobs'][index]['directory'] / 'rank0.json'
            row = p.read(file)
            row['operator_states'][0]['calls'] -= 1
            p.write(file, row)
            completed[index]['rank_sha256']['rank0.json'] = p.sha(file)
            p.write(root / 'completed-jobs.json', completed)
            with self.assertRaisesRegex(ValueError, 'actual attn projection call count'):
                reporting.replay(root)


if __name__ == '__main__':
    unittest.main()
