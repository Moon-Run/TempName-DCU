#!/usr/bin/env python3
"""Prepare, run and audit an isolated single-node BW1000 comparison."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
import protocol as p
import report as reporting

REPO = Path(__file__).resolve().parents[3]


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--blocks', type=int, default=10,
                        help='Five-arm blocks per trial; 1 is a short test, multiples of 10 are balanced')
    parser.add_argument('--execute', action='store_true', help='Without this flag, only print the plan')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--hipcc', help='Native HIP compiler; default discovered from target environment')
    parser.add_argument('--world-size', type=int, choices=(2, 4, 8), default=8)
    parser.add_argument('--chunk-rows', type=int)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--operator-warmup', type=int, default=100)
    parser.add_argument('--operator-iters', type=int, default=200)
    parser.add_argument('--quant-budget', type=float, default=.25,
                        help='Maximum fraction of physical blocks quantized by the selective arm')
    parser.add_argument('--calibration-iters', type=int, default=5)
    parser.add_argument('--calibration-pairs', type=int, default=3,
                        help='Independent paired repetitions in each of training and confirmation')
    parser.add_argument('--calibration-warmup', type=int, default=2)
    parser.add_argument('--calibration-min-gain', type=float, default=.002,
                        help='Minimum paired relative gain to admit a calibration candidate')
    scopes = parser.add_mutually_exclusive_group()
    scopes.add_argument('--operator-only', action='store_true')
    scopes.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Independent operator profile after timing')
    parser.add_argument('--small-model', action='store_true', help='Diagnostic 2-layer model, not GPT6.7B results')
    parser.add_argument('--worker-timeout', type=int, default=7200)
    parser.add_argument('--report', action='store_true', help='Read-only audit of a completed --out directory')
    args = parser.parse_args(argv)
    p.orders(args.trials, args.blocks)
    p.require(args.warmup_steps >= 0 and args.operator_warmup >= 0,
              'Warmup counts must be nonnegative')
    p.require(args.steps > 0 and args.operator_iters > 0 and args.worker_timeout > 0,
              'Timing counts and timeout must be positive')
    args.calibration = dict(warmup=args.calibration_warmup, iterations=args.calibration_iters,
        training_pairs=args.calibration_pairs, confirmation_pairs=args.calibration_pairs,
        min_gain=args.calibration_min_gain, quant_budget=args.quant_budget)
    p.validate_calibration_options(args.calibration)
    p.require(not args.report or (args.out is not None and not args.execute
              and not args.operator_only and not args.preflight_only),
              '--report requires --out and cannot execute a stage')
    model = dict(p.SMALL_MODEL if args.small_model else p.MODEL)
    if args.small_model:
        model['chunk_rows'] = model['sequence'] * model['micro_batch'] // args.world_size // 2
    if args.chunk_rows is not None:
        model['chunk_rows'] = args.chunk_rows
    p.validate_model(model, args.world_size)
    p.require(model['sequence'] * model['micro_batch'] // args.world_size // model['chunk_rows'] >= 2,
              'At least two blocks per destination are required to distinguish the two schedules')
    args.model = model
    if args.out is None:
        timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO / 'logs/dcu/phase6' / f'init-five-arm-{timestamp}'
    args.out = args.out.expanduser().resolve()
    args.scope = 'preflight' if args.preflight_only else 'operator' if args.operator_only else 'train'
    p.require(args.report or args.scope != 'train' or args.warmup_steps >= 1,
              'Full training requires at least one warmup step for per-module native checks')
    return args


def source_files():
    files = []
    for directory in ('python/flux_dcu', 'src', 'tools/dcu/phase6', 'third_party'):
        root = REPO / directory
        if not root.exists():
            continue
        for file in root.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix not in ('.pyc', '.so', '.o'):
                files.append(file)
    for name in ('setup.py', 'pyproject.toml', 'NOTICE', 'LICENSE', 'README.md'):
        file = REPO / name
        if file.is_file():
            files.append(file)
    return sorted(set(files))


def plan_for(args, manifest=None, run_id=None):
    config = dict(protocol=p.PROTOCOL, run_id=run_id or 'PREVIEW', world_size=args.world_size,
        model=args.model, timing=dict(warmup_steps=args.warmup_steps, timed_steps=args.steps,
            operator_warmup=args.operator_warmup, operator_iterations=args.operator_iters),
        seed=1234, token_seed=4321, hipcc=args.hipcc, build_dir=str(args.out / 'build'),
        profile=args.profile, source_manifest_sha256=p.digest(manifest) if manifest is not None else 'PREVIEW',
        recommended_image=p.IMAGE, backend=p.BACKEND,
        calibration=args.calibration, execution_plan=None, execution_plan_sha256=None)
    return dict(config=config, scope=args.scope, trials=args.trials, blocks=args.blocks,
                counterbalanced=p.balanced(args.blocks), schedule=p.orders(args.trials, args.blocks),
                jobs=p.jobs(args.trials, args.blocks, args.scope))


def command_for(python, source_root, directory, job, world):
    return [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        f'--nproc_per_node={world}', str(source_root / 'python/flux_dcu/worker.py'),
        '--config', str(directory / 'config.json'), '--out', str(directory),
        '--stage', job['stage'], '--arm', job['arm']]


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def launch(command, directory, source_root, timeout):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(source_root / 'python')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    # Preserve site RCCL/network/device settings. Never inject CUDA architecture
    # flags or NVIDIA-only defaults into a HIP process.
    with (directory / 'process.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=source_root, env=env, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            stop_process(process)
            raise
    if returncode:
        tail = (directory / 'process.log').read_text(errors='replace').splitlines()[-35:]
        raise RuntimeError(f"Worker exited {returncode}; {directory / 'process.log'}\n" + '\n'.join(tail))


def execute(args):
    p.require(not args.out.exists(), 'Output directory already exists; use a fresh --out')
    python = shutil.which(args.python)
    p.require(python is not None, f'Python executable not found: {args.python}')
    required = ('python/flux_dcu/worker.py', 'python/flux_dcu/runtime.py', 'src/taco/codec.hip')
    p.require(all((REPO / name).is_file() for name in required), 'Incomplete DCU source checkout')
    reporting.validate_reference_sources(REPO)
    files = source_files()
    manifest = {str(file.relative_to(REPO)): p.sha(file) for file in files}
    plan = plan_for(args, manifest=manifest, run_id=uuid.uuid4().hex)
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(protocol=p.PROTOCOL, run_id=plan['config']['run_id'], status='preparing',
                 completed=False, scope=args.scope, finished_jobs=0)
    p.write(args.out / 'batch-state.json', state)
    try:
        source_root = args.out / 'sources'
        for file in files:
            dest = source_root / file.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
            p.require(p.sha(dest) == manifest[str(file.relative_to(REPO))], 'Source changed while freezing')
        p.write(args.out / 'source-manifest.json', manifest)
        p.write(args.out / 'plan.json', plan)
        completed = []
        p.write(args.out / 'completed-jobs.json', completed)
        identities = None
        environment = None
        native_identity = None
        groups = {}
        execution_plan = None
        for job in plan['jobs']:
            directory = args.out / 'results' / job['directory']
            directory.mkdir(parents=True, exist_ok=False)
            job_config = p.config_for_job(plan['config'], job, execution_plan)
            p.write(directory / 'config.json', job_config)
            command = command_for(python, source_root, directory, job, args.world_size)
            p.write(directory / 'command.json', dict(job=job, command=command,
                environment={k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'RCCL_', 'HIP_', 'ROCR_', 'HSA_'))
                             or k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
            state.update(status='running', active_job=job)
            p.write(args.out / 'batch-state.json', state)
            print(f"[{job['index'] + 1}/{len(plan['jobs'])}] {job['directory']}", flush=True)
            launch(command, directory, source_root, args.worker_timeout)
            rows = p.validate_rows([p.read(directory / f'rank{rank}.json') for rank in range(args.world_size)],
                                   job, job_config)
            current_environment = [r['environment_identity'] for r in rows]
            p.require(environment is None or current_environment == environment,
                      'Hardware/software changed between workers')
            environment = current_environment
            current_native = [r['codec_build'] for r in rows]
            p.require(native_identity is None or native_identity == current_native,
                      'Native codec changed between workers')
            native_identity = current_native
            reporting.validate_artifacts(args.out, directory, rows, job_config)
            if job['stage'] == 'calibrate':
                p.require(execution_plan is None, 'Calibration must run once per fresh experiment')
                execution_plan = p.calibration_from_rows(rows, job_config)
                reporting.validate_calibration_evidence(args.out, directory, rows, job_config)
                p.write(args.out / 'selection-plan.json', execution_plan)
                state.update(execution_plan_sha256=p.digest(execution_plan))
            elif job['stage'] == 'train':
                identity = p.training_identity(rows)
                p.require(identities is None or identity == identities, 'Initial weights/input differ across windows')
                identities = identity
                group = groups.setdefault((job['trial'], job['block']), {})
                group[job['arm']] = rows
                for other in p.FULL_TACO_ARMS[1:]:
                    if 'flux_taco' in group and other in group:
                        p.compare_taco_trajectories(group['flux_taco'], group[other])
            done = dict(job=job, command_sha256=p.sha(directory / 'command.json'),
                rank_sha256={f'rank{rank}.json': p.sha(directory / f'rank{rank}.json')
                             for rank in range(args.world_size)})
            if job['stage'] == 'calibrate':
                done.update(execution_plan_sha256=p.digest(execution_plan),
                            selection_plan_file_sha256=p.sha(args.out / 'selection-plan.json'))
            completed.append(done)
            p.write(args.out / 'completed-jobs.json', completed)
            state.update(finished_jobs=len(completed))
            p.write(args.out / 'batch-state.json', state)
        report = reporting.replay(args.out)
        p.write(args.out / 'report.json', report)
        (args.out / 'report.md').write_text(reporting.markdown(report))
        state.update(status='completed' if args.scope == 'train' else args.scope + '_completed',
                     completed=True, active_job=None, report_sha256=p.sha(args.out / 'report.json'),
                     formal_optimizer_steps=report['formal_optimizer_steps'])
        p.write(args.out / 'batch-state.json', state)
        print(f"Completed scope: {args.scope}\nReport: {args.out / 'report.md'}", flush=True)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     completed=False, error=f'{type(exc).__name__}: {exc}')
        p.write(args.out / 'batch-state.json', state)
        raise


def main(argv=None):
    args = arguments(argv)
    if args.report:
        print(reporting.markdown(reporting.replay(args.out)))
    elif args.execute:
        # A scheduler/user SIGTERM should also clean up the worker's dedicated
        # process group, rather than orphaning torchrun after the controller exits.
        def terminate(_number, _frame):
            raise KeyboardInterrupt('Received SIGTERM')
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            execute(args)
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        import json
        print(json.dumps(dict(preview=True, writes_files=False, uses_gpu=False,
            output=str(args.out), **plan_for(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; only this controller\'s worker process group was terminated.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
