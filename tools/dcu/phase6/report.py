"""Read-only validation and paired statistics for a completed DCU experiment."""
import math
from pathlib import Path
import random
import statistics

import protocol as p


def pair_statistics(candidate, reference, balanced, bootstrap=10000):
    p.require(len(candidate) == len(reference) and candidate, 'Unpaired samples')
    logs = [math.log(x / y) for x, y in zip(candidate, reference)]
    estimate = 100 * (1 - math.exp(statistics.mean(logs)))
    interval = None
    if balanced and len(logs) >= 10:
        rng = random.Random(20261108)
        samples = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=len(logs)))))
                         for _ in range(bootstrap))
        interval = [samples[int((bootstrap - 1) * .025)], samples[int((bootstrap - 1) * .975)]]
    return dict(percent=estimate, ci95=interval, paired_blocks=len(logs),
                faster_blocks=sum(x < y for x, y in zip(candidate, reference)))


def summary(samples):
    return dict(mean=statistics.mean(samples), median=statistics.median(samples),
                min=min(samples), max=max(samples), windows=len(samples))


def summarize_stage(samples, stage, trials, blocks):
    result = {}
    roles = ('attn', 'mlp') if stage == 'operator' else ('full_step',)
    for trial in range(1, trials + 1):
        by_role = {}
        for role in roles:
            values = {arm: [next(s['value'] for s in samples if s['trial'] == trial
                                and s['block'] == block and s['arm'] == arm and s['role'] == role)
                           for block in range(blocks)] for arm in p.ARMS}
            pairs = {}
            for x, y in (('flux_taco', 'flux'), ('flux_taco_interleaved', 'flux_taco'),
                         (p.ARRIVAL, 'flux_taco_interleaved'), (p.SELECTIVE, p.ARRIVAL),
                         ('flux_taco_interleaved', 'flux'), (p.ARRIVAL, 'flux'), (p.SELECTIVE, 'flux')):
                pairs[f'{x}/{y}'] = pair_statistics(values[x], values[y], p.balanced(blocks))
            arm_summaries = {a: summary(v) for a, v in values.items()}
            for arm in p.ARMS:
                selected = [s for s in samples if s['trial'] == trial and s['arm'] == arm and s['role'] == role]
                if stage == 'train':
                    arm_summaries[arm].update(tokens_per_second=selected[0]['tokens_per_step'] * 1000 /
                        arm_summaries[arm]['mean'],
                        mean_training_loss=statistics.mean(s['mean_loss'] for s in selected),
                        mean_final_loss=statistics.mean(s['final_loss'] for s in selected),
                        mean_grad_norm=statistics.mean(s['mean_grad_norm'] for s in selected),
                        mean_final_grad_norm=statistics.mean(s['final_grad_norm'] for s in selected),
                        max_peak_allocated_gib=max(s['peak_allocated_bytes'] for s in selected) / 2**30,
                        max_peak_reserved_gib=max(s['peak_reserved_bytes'] for s in selected) / 2**30,
                        max_operator_scratch_gib=max(s['operator_scratch_bytes'] for s in selected) / 2**30)
                else:
                    arm_summaries[arm].update(
                        full_gemm_only_mean_us=statistics.mean(s['full_gemm_only_us'] for s in selected),
                        max_bf16_relative_l2=max(s['bf16_relative_l2'] for s in selected),
                        max_codec_reference_relative_l2=max(s['codec_reference_relative_l2'] for s in selected),
                        artificial_bypass_rank_windows=sum(len(s['artificial_bypass_ranks']) for s in selected),
                        artificial_diagnostic_rank_windows=sum(s['artificial_diagnostic_ranks'] for s in selected))
            by_role[role] = dict(unit='us/call' if stage == 'operator' else 'ms/step',
                                 arms=arm_summaries, pairs=pairs)
        result[f'trial{trial}'] = by_role
    return result


def validate_artifacts(root, directory, rows, config):
    """Validate actual native library and optional independent traces in-place.

    Report replay deliberately fails when native build artifacts were omitted
    from an archive; it does not silently claim to revalidate absent binaries.
    """
    import hashlib
    import json
    root, directory = Path(root), Path(directory)
    evidence = {}
    codec_source_sha = p.sha(root / 'sources/src/taco/codec.hip')
    for row in rows:
        build = row['codec_build']
        p.require(build['identity']['source_sha256'] == codec_source_sha,
                  'Native codec source differs from frozen source')
        key = hashlib.sha256(json.dumps(build['identity'], sort_keys=True).encode()).hexdigest()[:24]
        native = root / 'build' / key
        manifest = native / 'manifest.json'
        library = native / 'libflux_dcu_taco.so'
        p.require(p.read(manifest) == build, 'On-disk native build manifest changed')
        p.require(p.sha(library) == build['library_sha256'], 'On-disk native library bytes changed')
        evidence[str(manifest.relative_to(root))] = p.sha(manifest)
        evidence[str(library.relative_to(root))] = build['library_sha256']
        if row['stage'] == 'operator' and config['profile']:
            for record in row['operator_results']:
                diagnostic = record['profile']
                name = diagnostic['trace']
                p.require(Path(name).name == name, 'Profile trace must be inside its worker directory')
                file = directory / name
                p.require(p.sha(file) == diagnostic['trace_sha256'], 'Profile trace bytes changed')
                evidence[str(file.relative_to(root))] = diagnostic['trace_sha256']
    return evidence


def validate_reference_sources(source_root):
    """Cross-check retained upstream reference files against their provenance."""
    source_root = Path(source_root)
    definitions = (
        ('flux', 'provenance.json', 'commit', '19831ca2d820e3e782ed1d15d8b52d0898b78b26'),
        ('taco', 'manifest.json', 'source_revision', 'ca717e05b6dd6d21fff4d2617d209c95be63f3ef'),
    )
    references = {}
    for name, filename, revision_key, revision in definitions:
        directory = source_root / 'third_party' / name
        record = p.read(directory / filename)
        p.require(record.get(revision_key) == revision, f'Unexpected {name} reference revision')
        files = record.get('files')
        p.require(isinstance(files, dict) and files, f'Missing {name} reference file hashes')
        for relative, detail in files.items():
            path = Path(relative)
            p.require(not path.is_absolute() and '..' not in path.parts,
                      'Reference paths must stay inside third_party')
            expected = detail.get('sha256') if isinstance(detail, dict) else detail
            p.require(p.valid_hash(expected) and p.sha(directory / path) == expected,
                      f'Upstream reference bytes changed: {name}/{relative}')
        references[name] = record
    return references


def execution_plan_summary(bundle, config):
    if bundle is None:
        return None
    model, world = config['model'], config['world_size']
    base = [item['physical'] for item in p.expected_schedule(model, world, 'flux_taco_interleaved')]
    elements = model['chunk_rows'] * model['hidden']
    count = len(base)
    plain = (world - 1) * count * elements * 2
    full = (world - 1) * count * (elements // 128 * 136)
    roles = {}
    for role, plan in bundle['roles'].items():
        chosen = sum(plan['selective_mask'])
        wire = plain - (world - 1) * chosen * (elements // 128 * 120)
        roles[role] = dict(arrival_identity=plan['arrival_order'] == base,
            arrival_changed_slots=sum(a != b for a, b in zip(plan['arrival_order'], base)),
            selective_blocks=chosen, physical_blocks=count,
            actual_quantized_remote_fraction=chosen / count,
            remote_bytes_all_sources=wire, bf16_remote_bytes_all_sources=plain,
            full_quantized_remote_bytes_all_sources=full,
            remote_bytes_ratio_to_bf16=wire / plain,
            remote_bytes_ratio_to_full_quantized=wire / full,
            remote_bytes_reduction_vs_bf16_percent=100 * (1 - wire / plain),
            role_plan_sha256=p.digest(plan), diagnostics=plan['diagnostics'])
    return dict(execution_plan_sha256=p.digest(bundle), budget=config['calibration']['quant_budget'],
                transport_lanes=2, roles=roles,
                limitation='Arrival identity describes producer order only; D/E still use per-lane-ready consumers.')


def validate_calibration_evidence(root, directory, rows, config):
    """Re-fit the frozen planner from rank-local measurements without a GPU."""
    import importlib.util
    source = Path(root) / 'sources/python/flux_dcu/planning.py'
    spec = importlib.util.spec_from_file_location('dcu_frozen_planner_audit', source)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    plan = p.calibration_from_rows(rows, config)
    evidence = {}
    all_samples = []
    for row in rows:
        record = row['calibration_results']
        p.require(record.get('plan_sha256') == p.digest(plan), 'Calibration digest mismatch')
        samples = record.get('raw_samples')
        p.require(isinstance(samples, list) and samples, 'Missing raw calibration measurements')
        name = record.get('raw_samples_file')
        p.require(isinstance(name, str) and Path(name).name == name, 'Invalid calibration sample path')
        file = Path(directory) / name
        p.require(p.read(file) == dict(schema='dcu-calibration-samples-v2', rank=row['rank'], samples=samples),
                  'Calibration sample file differs from rank evidence')
        evidence[str(file.relative_to(root))] = p.sha(file)
        all_samples.append(samples)
    p.require(all(len(samples) == len(all_samples[0]) for samples in all_samples), 'Calibration rank sample count differs')
    for items in zip(*all_samples):
        p.require(all({k: v for k, v in item.items() if k != 'local_elapsed_seconds'} ==
                      {k: v for k, v in items[0].items() if k != 'local_elapsed_seconds'} for item in items),
                  'Calibration labels/inputs/aggregate times differ across ranks')
        p.require(all(p.finite_number(item.get('local_elapsed_seconds')) and item['local_elapsed_seconds'] > 0
                      for item in items), 'Invalid rank-local calibration time')
        p.require(items[0].get('max_rank_elapsed_seconds') == max(item['local_elapsed_seconds'] for item in items),
                  'Calibration timing must use actual maximum rank elapsed')
        for item in items:
            p.require(item.get('correctness_passed') is True
                      and p.finite_number(item.get('bf16_relative_l2'), True) and item['bf16_relative_l2'] <= .05
                      and p.finite_number(item.get('codec_reference_relative_l2'), True)
                      and item['codec_reference_relative_l2'] <= .01,
                      'Calibration candidate failed numerical gates')
    model, cfg, world = config['model'], config['calibration'], config['world_size']
    samples = iter(all_samples[0])
    for role_index, role in enumerate(('attn', 'mlp')):
        def evaluate(candidate, seed, arm, label):
            sample = next(samples, None)
            p.require(sample is not None, 'Missing calibration measurement during replay')
            expected = dict(role=role, label=label, seed=seed, arm=arm,
                plan_digest=planner.role_digest(candidate), arrival_order=candidate['arrival_order'],
                selective_mask=candidate['selective_mask'], iterations=cfg['iterations'], warmup=cfg['warmup'])
            p.require(all(sample.get(k) == v for k, v in expected.items()),
                      'Calibration candidate/seed/order does not match frozen planner replay')
            return sample['max_rank_elapsed_seconds']
        k = model['hidden' if role == 'attn' else 'ffn'] // world
        rebuilt = planner.fit_role(model['sequence'] * model['micro_batch'], model['hidden'], k, world,
            model['chunk_rows'], evaluate, config=cfg,
            seed=8000000 + 10000000 * role_index + config['seed'])
        p.require(rebuilt == plan['roles'][role], 'Measured calibration replay does not reproduce frozen role plan')
    p.require(next(samples, None) is None, 'Unexpected extra calibration samples')
    return evidence


def replay(root):
    root = Path(root)
    plan = p.read(root / 'plan.json')
    config = plan['config']
    p.require(config['protocol'] == p.PROTOCOL, 'Wrong protocol')
    p.validate_model(config['model'], config['world_size'])
    p.validate_calibration_options(config['calibration'])
    p.require(config.get('execution_plan') is None and config.get('execution_plan_sha256') is None,
              'The immutable base plan must not be rewritten after calibration')
    p.require(plan['jobs'] == p.jobs(plan['trials'], plan['blocks'], plan['scope']),
              'Frozen schedule changed')
    manifest = p.read(root / 'source-manifest.json')
    p.require(p.digest(manifest) == config['source_manifest_sha256'], 'Source manifest changed')
    for name, expected in manifest.items():
        p.require(p.sha(root / 'sources' / name) == expected, f'Frozen source changed: {name}')
    references = validate_reference_sources(root / 'sources')
    completed = p.read(root / 'completed-jobs.json')
    p.require(len(completed) == len(plan['jobs']), 'Incomplete worker sequence')
    samples = {'operator': [], 'train': []}
    trajectories = {}
    identities = None
    operator_identities = None
    environment = None
    native_identity = None
    rank_files = {}
    artifact_files = {}
    execution_plan = None
    calibration_rank_files = {}
    for job, done in zip(plan['jobs'], completed):
        p.require(job == done['job'], 'Completed job order/identity mismatch')
        directory = root / 'results' / job['directory']
        job_config = p.read(directory / 'config.json')
        p.require(job_config == p.config_for_job(config, job, execution_plan), 'Job configuration drift')
        command = p.read(directory / 'command.json')
        p.require(command.get('job') == job and p.sha(directory / 'command.json') == done['command_sha256'],
                  'Worker command identity drift')
        rows = []
        for rank in range(config['world_size']):
            name = f'rank{rank}.json'
            file = directory / name
            p.require(p.sha(file) == done['rank_sha256'][name], 'Rank result bytes changed')
            rank_files[str(file.relative_to(root))] = done['rank_sha256'][name]
            rows.append(p.read(file))
        rows = p.validate_rows(rows, job, job_config)
        current_environment = [r['environment_identity'] for r in rows]
        p.require(environment is None or environment == current_environment,
                  'Hardware/software identity changed between workers')
        environment = current_environment
        current_native = [r['codec_build'] for r in rows]
        p.require(native_identity is None or native_identity == current_native,
                  'Native codec changed between workers')
        native_identity = current_native
        artifact_files.update(validate_artifacts(root, directory, rows, job_config))
        if job['stage'] == 'calibrate':
            p.require(execution_plan is None, 'Repeated calibration in a frozen experiment')
            execution_plan = p.calibration_from_rows(rows, job_config)
            file = root / 'selection-plan.json'
            p.require(p.read(file) == execution_plan, 'Selection plan differs from calibration rank evidence')
            p.require(done.get('execution_plan_sha256') == p.digest(execution_plan)
                      and done.get('selection_plan_file_sha256') == p.sha(file),
                      'Frozen selection-plan hash changed')
            artifact_files['selection-plan.json'] = p.sha(file)
            calibration_rank_files = {str((directory / name).relative_to(root)): value
                                      for name, value in done['rank_sha256'].items()}
            artifact_files.update(validate_calibration_evidence(root, directory, rows, job_config))
        elif job['stage'] == 'train':
            identity = p.training_identity(rows)
            p.require(identities is None or identity == identities,
                      'Training initialization/input changed between windows')
            identities = identity
            trajectories.setdefault((job['trial'], job['block']), {})[job['arm']] = rows
            samples['train'].append(dict(trial=job['trial'], block=job['block'], arm=job['arm'],
                role='full_step', value=max(r['elapsed_seconds'] for r in rows) * 1000 /
                config['timing']['timed_steps'], tokens_per_step=rows[0]['tokens_per_step'],
                mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1],
                mean_grad_norm=statistics.mean(rows[0]['grad_norms']), final_grad_norm=rows[0]['grad_norms'][-1],
                peak_allocated_bytes=max(r['memory']['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['memory']['peak_reserved_bytes'] for r in rows),
                operator_scratch_bytes=max(sum(s['scratch_bytes'] for s in r['operator_states']) for r in rows)))
        elif job['stage'] == 'operator':
            current = p.training_identity(rows)
            p.require(operator_identities is None or current == operator_identities,
                      'Operator input/weights changed between windows')
            operator_identities = current
            for role in ('attn', 'mlp'):
                records = [next(v for v in r['operator_results'] if v['role'] == role) for r in rows]
                samples['operator'].append(dict(trial=job['trial'], block=job['block'], arm=job['arm'],
                    role=role, value=max(r['elapsed_seconds'] for r in records) * 1e6 /
                    config['timing']['operator_iterations'],
                    full_gemm_only_us=max(r['full_gemm_only_seconds'] for r in records) * 1e6 /
                    config['timing']['operator_iterations'],
                    bf16_relative_l2=max(r['bf16_relative_l2'] for r in records),
                    codec_reference_relative_l2=max(r['codec_reference_relative_l2'] for r in records),
                    artificial_bypass_ranks=[row['rank'] for row, record in zip(rows, records)
                        if record.get('arrival_diagnostic', {}) and record['arrival_diagnostic']['ready_bypass_observed']],
                    artificial_diagnostic_ranks=config['world_size'] if job['arm'] in (p.ARRIVAL, p.SELECTIVE) else 0))
    for group in trajectories.values():
        p.require(set(group) == set(p.ARMS), 'Incomplete five-arm training block')
        for other in p.FULL_TACO_ARMS[1:]:
            p.compare_taco_trajectories(group['flux_taco'], group[other])
    report = dict(protocol=p.PROTOCOL, run_id=config['run_id'], completed=True,
        completed_scope=plan['scope'], hardware_target='BW1000, single-node',
        standard_tp8_workload=(config['world_size'] == 8 and config['model'] == p.MODEL),
        implementation='Flux-design DCU chunked HIP/PyTorch port; not upstream CUDA CUTLASS Flux',
        backend=p.BACKEND, original_cuda_flux=False,
        cta_level_swizzle=False, environment_identity=environment,
        world_size=config['world_size'], model=config['model'], timing=config['timing'],
        recommended_image=p.IMAGE, trials=plan['trials'], blocks=plan['blocks'],
        counterbalanced=p.balanced(plan['blocks']),
        ci_available=p.balanced(plan['blocks']) and bool(samples['operator']),
        short_run=not p.balanced(plan['blocks']), all_windows_retained=True,
        correctness_passed=True, full_quantized_trajectories_identical=bool(trajectories),
        trajectory_equivalence_arms=list(p.FULL_TACO_ARMS),
        formal_optimizer_steps=len(samples['train']) * config['timing']['timed_steps'],
        training_windows=len(samples['train']), rank_sha256=rank_files, artifact_sha256=artifact_files,
        codec_build_by_rank=native_identity, profile_requested=config['profile'],
        reference_sources=references,
        calibration_config=config['calibration'], calibration_rank_sha256=calibration_rank_files,
        selection=execution_plan_summary(execution_plan, config),
        execution_plan_sha256=p.digest(execution_plan) if execution_plan is not None else None,
        actual_kernel_overlap_validated=False,
        samples=samples, results={})
    for stage in ('operator', 'train'):
        if samples[stage]:
            report['results'][stage] = summarize_stage(samples[stage], stage, plan['trials'], plan['blocks'])
    return report


def markdown(report):
    lines = ['# BW1000 五组实验', '',
        f"协议：`{report['protocol']}`；完成范围：`{report['completed_scope']}`。", '',
        '实现为 Flux 设计的 DCU 分块移植，不是原 CUDA/CUTLASS Flux 二进制。', '']
    if report['short_run']:
        lines += ['本次未完成十块平衡周期，仅用于短测；不计算置信区间。', '']
    lines += [f"完整训练窗口：{report['training_windows']}；累计正式 optimizer step："
              f"{report['formal_optimizer_steps']}（各窗口独立初始化）。", '']
    if report['selection']:
        lines += [f"独立校准计划：`{report['execution_plan_sha256']}`。量化预算上限："
                  f"{report['selection']['budget']:.4%}。", '',
                  '| 角色 | 到达生产顺序identity | 改变槽位 | 实际量化块比例 | 远端字节/BF16 | 远端字节/全量量化 |',
                  '| --- | --- | ---: | ---: | ---: | ---: |']
        for role, value in report['selection']['roles'].items():
            lines.append(f"| {role} | {value['arrival_identity']} | {value['arrival_changed_slots']} | "
                         f"{value['actual_quantized_remote_fraction']:.4%} | "
                         f"{value['remote_bytes_ratio_to_bf16']:.6f} | {value['remote_bytes_ratio_to_full_quantized']:.6f} |")
        lines += ['', 'identity只描述生产排程；D/E仍采用每lane就绪消费。零量化或identity回退如实保留。'
                  '校准不计入正式operator/训练窗口；每个trial共用本批冻结计划。', '']
    for stage, trials in report['results'].items():
        for trial, roles in trials.items():
            for role, result in roles.items():
                lines += [f"## {stage} / {trial} / {role}", '',
                    f"单位：{result['unit']}。每窗口取所有 rank 最大墙钟耗时。", '',
                    '| 组 | 均值 | 中位数 | 最慢窗口 |', '| --- | ---: | ---: | ---: |']
                for arm, values in result['arms'].items():
                    lines.append(f"| {arm} | {values['mean']:.6f} | {values['median']:.6f} | {values['max']:.6f} |")
                if stage == 'train':
                    lines += ['', '| 组 | tokens/s | 平均loss | 末步loss均值 | 平均grad norm | 末步grad norm均值 | peak allocated GiB | peak reserved GiB |',
                              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
                    for arm, values in result['arms'].items():
                        lines.append(f"| {arm} | {values['tokens_per_second']:.3f} | {values['mean_training_loss']:.8g} | "
                            f"{values['mean_final_loss']:.8g} | {values['mean_grad_norm']:.8g} | "
                            f"{values['mean_final_grad_norm']:.8g} | {values['max_peak_allocated_gib']:.4f} | "
                            f"{values['max_peak_reserved_gib']:.4f} |")
                else:
                    lines += ['', '| 组 | 独立完整GEMM均值 μs | 最大BF16相对L2 | 最大codec参考相对L2 |',
                              '| --- | ---: | ---: | ---: |']
                    for arm, values in result['arms'].items():
                        lines.append(f"| {arm} | {values['full_gemm_only_mean_us']:.6f} | "
                            f"{values['max_bf16_relative_l2']:.8g} | {values['max_codec_reference_relative_l2']:.8g} |")
                    lines += ['', '| 人工错峰诊断 | 观测到绕过的rank窗口 / 诊断rank窗口 |',
                              '| --- | ---: |']
                    for arm in (p.ARRIVAL, p.SELECTIVE):
                        values = result['arms'][arm]
                        lines.append(f"| {arm} | {values['artificial_bypass_rank_windows']}/{values['artificial_diagnostic_rank_windows']} |")
                    lines += ['', '该诊断使用独立实例与人工lane延迟，不进入正式计时。未观测到绕过也如实报告；'
                              '观测到绕过不代表普通负载获得加速。']
                lines += ['', '| 前者 / 后者 | 配对下降 % | 95% CI | 更快块数 |',
                          '| --- | ---: | --- | ---: |']
                for pair, value in result['pairs'].items():
                    interval = ('不计算' if value['ci95'] is None else
                                f"[{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")
                    lines.append(f"| {pair} | {value['percent']:+.4f} | {interval} | "
                                 f"{value['faster_blocks']}/{value['paired_blocks']} |")
                lines.append('')
    lines += ['正下降率表示前者更快。CI按完整块bootstrap，未做多重比较校正；'
              'rank与窗口内step不作为独立样本。不同trial分别报告。', '',
              'loss和grad norm仅取rank0的正式步，不重复计TP rank；末步值先取每窗口最后一步，再跨窗口平均。'
              'operator scratch属于Torch分配，已包含于allocator峰值，不重复相加；allocated与reserved也不相加。', '',
              '完整GEMM独立计时不含通信/归约，不能用总投影与它的比值单独归因分块损失。'
              '可选profile已核对文件哈希；通信event区间包含等待，报告不据此声称已验证kernel重叠。', '',
              '配置、源码快照、命令、逐rank结果、失败信息和可选profile保存在批次目录。', '']
    return '\n'.join(lines)
