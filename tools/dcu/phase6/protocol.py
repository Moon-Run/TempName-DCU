"""CPU-only contracts for calibrated, five-arm BW1000 experiments."""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'bw1000-flux-dcu-five-arm-arrival-selective-v2'
BACKEND = 'dcu-blas-rccl-block-lanes-v2'
ARRIVAL = 'flux_taco_interleaved_arrival'
SELECTIVE = 'flux_taco_interleaved_arrival_selective'
ARMS = ('flux', 'flux_taco', 'flux_taco_interleaved', ARRIVAL, SELECTIVE)
FULL_TACO_ARMS = ('flux_taco', 'flux_taco_interleaved', ARRIVAL)
PLAN_SCHEMA = 'dcu-arrival-selective-plan-v2'
CALIBRATION = dict(warmup=2, iterations=5, training_pairs=3,
                   confirmation_pairs=3, min_gain=.002, quant_budget=.25)
IMAGE = ('42.228.13.241:5000/jenkins/model_test_env/vllm:'
         '0.28.1rc1-ubuntu22.04-dtk26041-py3.10-20261019-1402')
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32,
             sequence=2048, micro_batch=1, accumulation_steps=4,
             vocab=33792, chunk_rows=128)
SMALL_MODEL = dict(layers=2, hidden=256, ffn=1024, heads=8,
                   sequence=128, micro_batch=1, accumulation_steps=1,
                   vocab=1024, chunk_rows=8)


def require(value, message):
    if not value:
        raise ValueError(message)


def positive_int(value):
    return type(value) is int and value > 0


def validate_model(model, world):
    require(world in (2, 4, 8), 'world_size must be 2, 4 or 8')
    require(set(model) == set(MODEL), 'Unexpected/missing model dimensions')
    require(all(positive_int(v) for v in model.values()), 'Positive integer dimensions required')
    require(model['hidden'] % model['heads'] == 0, 'hidden must be divisible by heads')
    require(all(model[k] % world == 0 for k in ('hidden', 'ffn', 'heads', 'vocab')),
            'TP dimensions must be divisible by world_size')
    rows = model['sequence'] * model['micro_batch']
    require(rows % world == 0 and (rows // world) % model['chunk_rows'] == 0,
            'chunk_rows must divide the per-rank output rows')
    require(model['hidden'] % 128 == 0, 'H128 codec requires hidden divisible by 128')
    return model


def orders(trials, blocks):
    require(positive_int(trials) and positive_int(blocks), 'trials and blocks must be positive')
    rotations = [list(ARMS[offset:] + ARMS[:offset]) for offset in range(len(ARMS))]
    permutations = rotations + [row[::-1] for row in rotations]
    result = []
    for trial in range(trials):
        cycle = permutations if trial % 2 == 0 else permutations[::-1]
        for block in range(blocks):
            result.append(dict(trial=trial + 1, block=block,
                               arms=list(cycle[block % len(cycle)])))
    return result


def jobs(trials, blocks, scope):
    require(scope in ('preflight', 'operator', 'train'), 'Invalid experiment scope')
    result = [dict(stage='preflight', arm='flux', trial=0, block=0,
                   index=0, directory='preflight')]
    if scope == 'preflight':
        return result
    result.append(dict(stage='calibrate', arm=ARRIVAL, trial=0, block=0,
                       index=1, directory='calibrate'))
    for stage in ('operator', 'train'):
        if stage == 'train' and scope != 'train':
            break
        for block in orders(trials, blocks):
            for arm in block['arms']:
                result.append(dict(stage=stage, arm=arm, trial=block['trial'],
                    block=block['block'], index=len(result),
                    directory=f"{stage}/trial{block['trial']}/block{block['block']:03d}-{arm}"))
    return result


def balanced(blocks):
    return blocks >= 10 and blocks % 10 == 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def finite_number(value, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_calibration_options(options):
    require(set(options) == set(CALIBRATION), 'Wrong calibration option keys')
    for key in ('iterations', 'training_pairs', 'confirmation_pairs'):
        require(positive_int(options[key]), f'Positive calibration {key} required')
    require(type(options['warmup']) is int and options['warmup'] >= 0, 'Invalid calibration warmup')
    require(finite_number(options['quant_budget']) and 0 <= options['quant_budget'] <= 1,
            'quant_budget must be in [0,1]')
    require(finite_number(options['min_gain']) and 0 <= options['min_gain'] < 1,
            'calibration min_gain must be in [0,1)')


def validate_execution_plan(bundle, config):
    model, world = config['model'], config['world_size']
    require(isinstance(bundle, dict) and bundle.get('schema') == PLAN_SCHEMA
            and bundle.get('world_size') == world and bundle.get('transport_lanes') == 2,
            'Wrong calibrated execution-plan identity')
    require(set(bundle.get('roles', {})) == {'attn', 'mlp'}, 'Missing calibrated role')
    count = model['sequence'] * model['micro_batch'] // model['chunk_rows']
    for role, plan in bundle['roles'].items():
        shape = [model['sequence'] * model['micro_batch'], model['hidden'],
                 model['hidden' if role == 'attn' else 'ffn'] // world]
        require(plan.get('shape') == shape and plan.get('chunk_rows') == model['chunk_rows'],
                'Calibrated plan shape/chunk mismatch')
        order, mask = plan.get('arrival_order'), plan.get('selective_mask')
        require(isinstance(order, list) and len(order) == count
                and all(type(i) is int for i in order) and sorted(order) == list(range(count)),
                'Arrival order must be a physical-block bijection')
        require(all(physical // (count // world) == slot % world for slot, physical in enumerate(order)),
                'Arrival plan must preserve interleaved destination slots')
        require(isinstance(mask, list) and len(mask) == count and all(type(x) is bool for x in mask),
                'Selection mask must contain one boolean per physical block')
        require(sum(mask) <= math.floor(count * config['calibration']['quant_budget'] + 1e-12),
                'Calibration selection exceeds frozen block budget')
        require(isinstance(plan.get('diagnostics'), dict), 'Missing calibration diagnostics')
    digest(bundle)  # Reject nonfinite diagnostics before writing or execution.
    return bundle


def config_for_job(base, job, execution_plan=None):
    require(base.get('execution_plan') is None and base.get('execution_plan_sha256') is None,
            'Base plan must remain immutable and uncalibrated')
    result = dict(base, job=job)
    if job['stage'] in ('operator', 'train'):
        validate_execution_plan(execution_plan, base)
        result.update(execution_plan=execution_plan, execution_plan_sha256=digest(execution_plan))
    return result


def calibration_from_rows(rows, config):
    plans = [row.get('calibration_plan') for row in rows]
    require(plans and all(plan == plans[0] for plan in plans), 'Ranks disagree on calibrated execution plan')
    return validate_execution_plan(plans[0], config)


def expected_schedule(model, world, arm, role_plan=None):
    per_destination = model['sequence'] * model['micro_batch'] // world // model['chunk_rows']
    result = []
    for slot in range(world * per_destination):
        if arm in (ARRIVAL, SELECTIVE):
            physical = role_plan['arrival_order'][slot]
            destination, within = divmod(physical, per_destination)
        else:
            destination, within = ((slot % world, slot // world) if arm == 'flux_taco_interleaved'
                                   else divmod(slot, per_destination))
            physical = destination * per_destination + within
        result.append(dict(slot=slot, destination=destination, within=within,
                           physical=physical, row=physical * model['chunk_rows'], rows=model['chunk_rows'],
                           lane=within % 2))
    return result


def validate_operator_state(state, row, model, world, role, calls, execution_plan=None):
    shape = [model['sequence'] * model['micro_batch'], model['hidden'],
             model['hidden' if role == 'attn' else 'ffn'] // world]
    role_plan = execution_plan['roles'][role] if row['arm'] in (ARRIVAL, SELECTIVE) else None
    schedule = expected_schedule(model, world, row['arm'], role_plan)
    mapping_sha = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    mask = (role_plan['selective_mask'] if row['arm'] == SELECTIVE else
            [row['arm'] != 'flux'] * len(schedule))
    per_destination = len(schedule) // world
    selected = [sum(flag and physical // per_destination != source for physical, flag in enumerate(mask))
                for source in range(world)]
    remote_blocks = (world - 1) * per_destination
    block_elements = model['chunk_rows'] * model['hidden']
    wires = [q * (block_elements // 128 * 136) + (remote_blocks - q) * block_elements * 2 for q in selected]
    require(state.get('backend') == BACKEND
            and state.get('arm') == row['arm'] and state.get('rank') == row['rank'],
            'Wrong operator backend/arm/rank')
    require(state.get('shape') == shape and state.get('chunk_rows') == model['chunk_rows'],
            'Wrong output-projection shape/chunk size')
    require(state.get('calls') == calls, f'Wrong actual {role} projection call count')
    require(state.get('production_schedule') == schedule and state.get('mapping_sha256') == mapping_sha,
            'Actual production mapping is not the frozen arm schedule')
    require(state.get('gemms_per_call') == len(schedule) and state.get('collectives_per_call') == len(schedule),
            'Wrong GEMM/collective count per projection')
    quantization = ('selective_remote_h128_e4m3' if row['arm'] == SELECTIVE else
                    'none' if row['arm'] == 'flux' else 'all_remote_h128_e4m3')
    require(state.get('quantization') == quantization
            and state.get('reduction_order') == list(range(world))
            and state.get('rounding') == 'bf16_after_each_source', 'Codec/reduction contract changed')
    require(state.get('remote_wire_bytes_per_source') == wires[row['rank']]
            and state.get('actual_remote_wire_bytes_per_rank') == wires[row['rank']]
            and state.get('remote_wire_bytes_by_source') == wires
            and state.get('selected_remote_blocks_by_source') == selected
            and state.get('selection_mask') == mask,
            'Remote wire-byte accounting mismatch')
    require(state.get('plan_sha256') == (digest(role_plan) if role_plan is not None else None),
            'Actual role-plan hash changed')
    require(state.get('original_cuda_flux') is False and state.get('cta_level_swizzle') is False,
            'DCU port cannot be reported as original CUDA/CTA execution')
    require(type(state.get('scratch_bytes')) is int and state['scratch_bytes'] >= 0,
            'Invalid scratch-byte accounting')
    require(state.get('shared_streams_per_group') == 5 and state.get('transport_lanes') == 2
            and state.get('consumer_policy') == ('per_lane_ready' if row['arm'] in (ARRIVAL, SELECTIVE) else 'ordered')
            and state.get('validation_calls') == (2 if row['stage'] == 'operator' else 1),
            'Missing untimed per-instance validation or unexpected stream policy')
    require(state.get('diagnostic_delay_cycles') == 0
            and state.get('diagnostic_delay_in_normal_forward') is False,
            'Artificial lane delay must not enter measured operators')


def validate_arrival_diagnostic(record, row, config):
    diagnostic = record.get('arrival_diagnostic')
    if row['arm'] not in (ARRIVAL, SELECTIVE):
        require(diagnostic is None, 'Unexpected arrival diagnostic on a control arm')
        return
    require(isinstance(diagnostic, dict) and diagnostic.get('passed') is True
            and diagnostic.get('numerical_bitwise_equal') is True
            and diagnostic.get('artificial') is True and diagnostic.get('ordinary_speedup_claim') is False,
            'Missing successful independent artificial lane-skew numerical check')
    require(diagnostic.get('delayed_rank') == 0 and diagnostic.get('delayed_lane') == 0
            and diagnostic.get('delay_cycles') == 50000000,
            'Artificial lane-skew protocol changed')
    base = expected_schedule(config['model'], config['world_size'], 'flux_taco_interleaved')
    slots = {item['physical']: item['slot'] for item in base}
    pending = base[0]['physical']
    require(diagnostic.get('delayed_physical') == pending, 'Wrong delayed physical block')
    events = diagnostic.get('evidence', {}).get('events')
    require(isinstance(events, list) and events, 'Missing independent lane-skew event evidence')
    for event in events:
        require(event.get('stage') in ('producer', 'communication', 'consumer')
                and type(event.get('physical')) is int and event['physical'] in slots
                and event.get('lane') in (0, 1)
                and finite_number(event.get('start_ms')) and finite_number(event.get('end_ms'))
                and event['end_ms'] >= event['start_ms'], 'Invalid lane-skew event interval')
    pending_events = [event for event in events if event['stage'] == 'communication' and event['physical'] == pending]
    require(len(pending_events) == 1, 'Missing/duplicate delayed communication event')
    overtakes = [event['physical'] for event in events if event['stage'] == 'consumer'
                and event['lane'] != 0 and slots[event['physical']] > slots[pending]
                and event['end_ms'] < pending_events[0]['end_ms']]
    require(diagnostic.get('later_blocks_completed_before_pending') == overtakes
            and diagnostic.get('ready_bypass_observed') is bool(overtakes),
            'Artificial bypass claim disagrees with receiver-local event evidence')


def expected_parameter_counts(model, world):
    h, f, v, s, layers = (model[k] for k in ('hidden', 'ffn', 'vocab', 'sequence', 'layers'))
    logical = v * h + s * h + layers * (4 * h * h + 2 * h * f + 9 * h + f) + 2 * h
    replicated = s * h + layers * 6 * h + 2 * h
    sharded = logical - replicated
    return dict(logical_parameters=logical, replicated_parameters=replicated,
                sharded_parameters=sharded, parameters_per_rank=sharded // world + replicated,
                tied_embedding_head=True)


def validate_rows(rows, job, config):
    world = config['world_size']
    require(len(rows) == world and {r.get('rank') for r in rows} == set(range(world)),
            'Missing, duplicate or unexpected rank')
    for row in rows:
        require(row.get('world_size') == world and row.get('stage') == job['stage']
                and row.get('arm') == job['arm'], 'Worker identity does not match job')
        require(row.get('completed') is True and row.get('passed') is True,
                f"Rank {row.get('rank')} did not complete correctness checks")
        require(row.get('run_id') == config['run_id'], 'Worker run_id mismatch')
        require(isinstance(row.get('environment_identity'), dict)
                and row['environment_identity'], 'Missing stable hardware/software identity')
        require(row.get('backend') == BACKEND
                and row.get('original_cuda_flux') is False and row.get('cta_level_swizzle') is False,
                'Wrong worker implementation identity')
        build = row.get('codec_build', {})
        require(valid_hash(build.get('library_sha256'))
                and valid_hash(build.get('identity', {}).get('source_sha256')),
                'Missing native codec provenance')
        require(row.get('execution_plan_sha256') == config.get('execution_plan_sha256'),
                'Worker execution plan hash changed')
        if job['stage'] == 'train':
            model = config['model']
            steps = config['timing']['timed_steps']
            require(row.get('timed_steps') == steps, 'Wrong training step denominator')
            require(row.get('warmup_steps') == config['timing']['warmup_steps'], 'Wrong training warmup count')
            require(finite_number(row.get('elapsed_seconds')) and row['elapsed_seconds'] > 0,
                    'Invalid training elapsed time')
            for key in ('losses', 'grad_norms'):
                values = row.get(key)
                require(isinstance(values, list) and len(values) == steps
                        and all(finite_number(v, True) for v in values),
                        f'Incomplete/nonfinite {key}')
            for key in ('initialization_sha256', 'input_sha256', 'final_parameters_sha256'):
                require(valid_hash(row.get(key)), f'Missing/invalid {key}')
            actual_model = row.get('actual_model_config', {})
            require(all(actual_model.get(k) == v for k, v in model.items()), 'Actual training model changed')
            counts = expected_parameter_counts(model, world)
            require(row.get('parameter_counts') == counts
                    and row.get('actual_parameters_per_rank') == counts['parameters_per_rank'],
                    'Wrong trainable parameter count')
            tokens = model['sequence'] * model['micro_batch'] * model['accumulation_steps']
            require(row.get('tokens_per_step') == tokens and row.get('dtype') == 'bfloat16'
                    and row.get('optimizer') == 'AdamW-fp32-master', 'Training dtype/optimizer/token budget changed')
            forwards = steps * model['accumulation_steps']
            require(row.get('operator_counters') == dict(model_forwards=forwards,
                calls_per_projection=forwards, attention_projection_calls=forwards * model['layers'],
                mlp_projection_calls=forwards * model['layers'], target_modules=2 * model['layers'],
                actual_calls_per_instance=[forwards] * (2 * model['layers']),
                actual_total_projection_calls=forwards * 2 * model['layers'],
                fallback_calls=0), 'Wrong formal model/module coverage or fallback')
            states = row.get('operator_states', [])
            require(len(states) == 2 * model['layers'], 'Missing actual operator state')
            for index, state in enumerate(states):
                validate_operator_state(state, row, model, world, 'attn' if index % 2 == 0 else 'mlp',
                    (steps + config['timing']['warmup_steps']) * model['accumulation_steps'], config['execution_plan'])
            memory = row.get('memory', {})
            require(all(type(memory.get(k)) is int and memory[k] >= 0 for k in
                        ('peak_allocated_bytes', 'peak_reserved_bytes')), 'Missing valid allocator peak memory')
        elif job['stage'] == 'operator':
            records = row.get('operator_results', [])
            require(len(records) == 2 and {r.get('role') for r in records} == {'attn', 'mlp'},
                    'Need both output-projection roles')
            for record in records:
                require(record.get('correctness_passed') is True,
                        'Independent operator correctness check failed')
                require(record.get('timed_iterations') == config['timing']['operator_iterations'],
                        'Wrong operator timing denominator')
                require(finite_number(record.get('elapsed_seconds'))
                        and record['elapsed_seconds'] > 0, 'Invalid operator elapsed time')
                for key, limit in (('bf16_relative_l2', .01 if row['arm'] == 'flux' else .05),
                                   ('codec_reference_relative_l2', .01), ('changed_input_relative_l2', .01)):
                    require(finite_number(record.get(key), True) and record[key] <= limit,
                            f'Operator numerical gate failed: {key}')
                require(row['arm'] == 'flux' or record.get('order_bitwise_equal') is True,
                        'TACO ordering-only numerical equivalence not verified')
                validate_arrival_diagnostic(record, row, config)
                require(finite_number(record.get('full_gemm_only_seconds'))
                        and record['full_gemm_only_seconds'] > 0, 'Missing full-GEMM control timing')
                validate_operator_state(record.get('execution', {}), row, config['model'], world, record['role'],
                    2 + int(bool(config['profile'])) + config['timing']['operator_warmup']
                    + config['timing']['operator_iterations'], config['execution_plan'])
                profile = record.get('profile')
                require(bool(profile) == bool(config['profile']), 'Requested independent profile missing/unexpected')
                if profile:
                    require(valid_hash(profile.get('trace_sha256')) and isinstance(profile.get('trace'), str),
                            'Missing profile trace identity')
            for key in ('initialization_sha256', 'input_sha256'):
                require(valid_hash(row.get(key)), f'Missing operator {key}')
        elif job['stage'] == 'calibrate':
            validate_execution_plan(row.get('calibration_plan'), config)
            require(isinstance(row.get('calibration_results'), dict) and row['calibration_results'],
                    'Missing rank-local calibration measurement evidence')
    if job['stage'] == 'calibrate':
        calibration_from_rows(rows, config)
    return sorted(rows, key=lambda r: r['rank'])


def training_identity(rows):
    return [[r['initialization_sha256'], r['input_sha256']] for r in rows]


def compare_taco_trajectories(left, right):
    require(training_identity(left) == training_identity(right),
            'TACO arms have different initialization/input')
    for a, b in zip(left, right):
        for key in ('losses', 'grad_norms'):
            require(a[key] == b[key], f'TACO ordering changed {key}; investigate before timing claims')
        if 'final_parameters_sha256' in a or 'final_parameters_sha256' in b:
            require(a.get('final_parameters_sha256') == b.get('final_parameters_sha256')
                    and a.get('final_parameters_sha256'), 'TACO final parameters differ')
