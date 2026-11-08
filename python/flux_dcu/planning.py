"""CPU-only plans and measured selection for arrival / closed-block quantization.

No arrival timestamp is interpreted as a predicted speedup. Candidate selection
uses complete-operator paired measurements; confirmation never reselects a plan.
"""
import hashlib
import json
import math
import statistics

SCHEMA = 'dcu-arrival-selective-plan-v2'
ARRIVAL_ARM = 'flux_taco_interleaved_arrival'
SELECTIVE_ARM = 'flux_taco_interleaved_arrival_selective'
DEFAULTS = dict(warmup=2, iterations=5, training_pairs=3, confirmation_pairs=3,
                min_gain=.002, quant_budget=.25)


def role_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _shape(m, n, k, world, chunk_rows):
    if any(type(x) is not int or x <= 0 for x in (m, n, k, world, chunk_rows)):
        raise ValueError('Positive integer shape/world/chunk_rows required')
    if world < 2 or m % (world * chunk_rows) or n % 128:
        raise ValueError('Require M divisible by world*chunk_rows and N by 128')
    return m // world // chunk_rows


def base_order(m, n, k, world, chunk_rows):
    per = _shape(m, n, k, world, chunk_rows)
    return [(slot % world) * per + slot // world for slot in range(per * world)]


def validate_role_plan(plan, m=None, n=None, k=None, world=None, chunk_rows=None,
                       *, shape=None, world_size=None):
    if shape is not None:
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError('shape must be [M,N,K]')
        if any(x is not None for x in (m, n, k)):
            raise ValueError('Use shape or individual M/N/K, not both')
        m, n, k = shape
    if world_size is not None:
        if world is not None and world != world_size:
            raise ValueError('Conflicting world/world_size')
        world = world_size
    per = _shape(m, n, k, world, chunk_rows)
    if not isinstance(plan, dict) or set(plan) != {'shape', 'chunk_rows', 'arrival_order',
                                                 'selective_mask', 'diagnostics'}:
        raise ValueError('Unexpected/missing role-plan fields')
    if plan['shape'] != [m, n, k] or plan['chunk_rows'] != chunk_rows:
        raise ValueError('Plan shape/chunk_rows differs from the actual operator')
    order, mask = plan['arrival_order'], plan['selective_mask']
    blocks = per * world
    if (not isinstance(order, list) or len(order) != blocks or
            any(type(v) is not int for v in order) or sorted(order) != list(range(blocks))):
        raise ValueError('Arrival order must be a bijection of physical blocks')
    if any(physical // per != slot % world for slot, physical in enumerate(order)):
        raise ValueError('Arrival order must preserve every interleaved destination slot')
    if not isinstance(mask, list) or len(mask) != blocks or any(type(v) is not bool for v in mask):
        raise ValueError('Selective mask must contain one bool per physical output block')
    if not isinstance(plan['diagnostics'], dict):
        raise ValueError('Plan diagnostics must be a dictionary')
    # Also reject NaN/Infinity or non-JSON contents before freezing identity.
    role_digest(plan)
    return plan


def make_role(m, n, k, world, chunk_rows, order=None, mask=None):
    baseline = base_order(m, n, k, world, chunk_rows)
    result = dict(shape=[m, n, k], chunk_rows=chunk_rows,
                  arrival_order=list(baseline if order is None else order),
                  selective_mask=list([True] * len(baseline) if mask is None else mask),
                  diagnostics={})
    return validate_role_plan(result, m, n, k, world, chunk_rows)


def candidates(m, n, k, world, chunk_rows):
    per = _shape(m, n, k, world, chunk_rows)
    originals = list(range(per))
    modes = [('identity', lambda destination: originals),
             ('reverse_all', lambda destination: originals[::-1]),
             ('reverse_odd', lambda destination: originals[::-1] if destination % 2 else originals),
             ('reverse_even', lambda destination: originals if destination % 2 else originals[::-1]),
             ('rotate', lambda destination: originals[1:] + originals[:1])]
    result, seen = [], set()
    for name, transform in modes:
        by_destination = [transform(d) for d in range(world)]
        order = [d * per + by_destination[d][within] for within in range(per) for d in range(world)]
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            result.append(dict(name=name, arrival_order=order))
    return result


def quantized_blocks(plan):
    return [i for i, selected in enumerate(plan['selective_mask']) if selected]


def quantization_statistics(plan, world):
    m, n, k = plan['shape']
    per = _shape(m, n, k, world, plan['chunk_rows'])
    validate_role_plan(plan, m, n, k, world, plan['chunk_rows'])
    selected = quantized_blocks(plan)
    selected_per_source = [sum(physical // per != source for physical in selected)
                           for source in range(world)]
    elements = plan['chunk_rows'] * n
    bf16_per_block, fp8_per_block = elements * 2, elements // 128 * 136
    remote_blocks = (world - 1) * per
    wire = [remote_blocks * bf16_per_block - count * (bf16_per_block - fp8_per_block)
            for count in selected_per_source]
    return dict(selection_unit='physical-output-macroblock-all-remote-sources',
                total_output_blocks=per * world, quantized_output_blocks=len(selected),
                selected_physical_blocks=selected,
                total_remote_contributions=(world - 1) * per * world,
                quantized_remote_contributions=(world - 1) * len(selected),
                actual_remote_fraction=len(selected) / (per * world),
                selected_per_source=selected_per_source,
                remote_wire_bytes_per_source=wire,
                bf16_remote_wire_bytes_per_source=[remote_blocks * bf16_per_block] * world,
                full_quantized_remote_wire_bytes_per_source=[remote_blocks * fp8_per_block] * world,
                mixed_source_blocks=0)


def settings(values=None):
    result = dict(DEFAULTS)
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(DEFAULTS):
            raise ValueError('Unknown calibration settings')
        result.update(values)
    for key in ('warmup', 'iterations', 'training_pairs', 'confirmation_pairs'):
        lower = 0 if key == 'warmup' else 1
        if type(result[key]) is not int or result[key] < lower:
            raise ValueError(f'Invalid calibration {key}')
    for key in ('min_gain', 'quant_budget'):
        value = result[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid calibration {key}')
        result[key] = float(value)
    if result['min_gain'] >= 1:
        raise ValueError('min_gain must be less than one')
    return result


def _measurement(value):
    if isinstance(value, dict) and value.get('correctness_passed', True) is not True:
        raise ValueError('Numerical failure in a calibration candidate; do not select or hide it')
    elapsed = value.get('elapsed_seconds') if isinstance(value, dict) else value
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Evaluator must return a positive complete-operator elapsed time')
    return float(elapsed)


def paired_measure(reference, candidate, evaluate, *, arm, pairs, seed, phase, name):
    """Evaluate AB/BA pairs on identical inputs, never pass a heldout score back.

    evaluate(plan, seed, arm, label) returns max-rank elapsed time (or a dict
    containing it). Fresh seed ranges separate fit and confirmation inputs.
    """
    records = []
    for pair in range(pairs):
        inputs = seed + pair
        order = [('reference', reference), ('candidate', candidate)]
        if pair % 2:
            order.reverse()
        measured = {}
        for side, plan in order:
            label = f'{phase}/{name}/pair{pair}/{side}'
            measured[side] = _measurement(evaluate(plan, inputs, arm, label))
        records.append(dict(pair=pair, seed=inputs, execution_order=[x[0] for x in order],
                            reference_seconds=measured['reference'],
                            candidate_seconds=measured['candidate'],
                            fractional_gain=1 - measured['candidate'] / measured['reference']))
    gain = 1 - math.exp(statistics.mean(math.log(r['candidate_seconds'] / r['reference_seconds'])
                                       for r in records))
    return dict(name=name, phase=phase, paired_samples=records, fractional_gain=gain,
                worst_pair_gain=min(r['fractional_gain'] for r in records),
                candidate_digest=role_digest(candidate), reference_digest=role_digest(reference))


def confirmed(record, minimum_gain):
    # A small pilot is not a confidence interval. Every independent confirmation
    # pair must exceed the predeclared threshold; no trimming or new candidate.
    return (record['fractional_gain'] > minimum_gain and
            all(r['fractional_gain'] > minimum_gain for r in record['paired_samples']))


def fit_role(m, n, k, world, chunk_rows, evaluate, *, config=None, seed=8000000):
    """Pure orchestration with injectable timing; returns one frozen role plan."""
    cfg = settings(config)
    initial = make_role(m, n, k, world, chunk_rows)
    alternatives = candidates(m, n, k, world, chunk_rows)
    order_trials = []
    best_order, best_name, best_gain = initial, 'identity', 0.
    for candidate in alternatives[1:]:
        plan = make_role(m, n, k, world, chunk_rows, candidate['arrival_order'])
        record = paired_measure(initial, plan, evaluate, arm=ARRIVAL_ARM,
            pairs=cfg['training_pairs'], seed=seed, phase='arrival_training', name=candidate['name'])
        order_trials.append(record)
        if record['fractional_gain'] > max(cfg['min_gain'], best_gain):
            best_order, best_name, best_gain = plan, candidate['name'], record['fractional_gain']
    order_confirmation = None
    order_accepted = False
    if best_name != 'identity':
        order_confirmation = paired_measure(initial, best_order, evaluate, arm=ARRIVAL_ARM,
            pairs=cfg['confirmation_pairs'], seed=seed + 1000000,
            phase='arrival_confirmation', name=best_name)
        order_accepted = confirmed(order_confirmation, cfg['min_gain'])
    if not order_accepted:
        best_order = initial
    fixed_order = best_order['arrival_order']
    blocks = len(fixed_order)
    zero = make_role(m, n, k, world, chunk_rows, fixed_order, [False] * blocks)
    all_quant = make_role(m, n, k, world, chunk_rows, fixed_order, [True] * blocks)
    cap = math.floor(blocks * cfg['quant_budget'])
    singleton_trials = []
    positive = []
    # A zero budget is a genuine Q0 plan; full quantization remains a control.
    if cap:
        for physical in range(blocks):
            mask = [False] * blocks; mask[physical] = True
            proposal = make_role(m, n, k, world, chunk_rows, fixed_order, mask)
            record = paired_measure(zero, proposal, evaluate, arm=SELECTIVE_ARM,
                pairs=cfg['training_pairs'], seed=seed + 2000000,
                phase='selective_singleton_training', name=f'block{physical}')
            record['physical_block'] = physical
            singleton_trials.append(record)
            if record['fractional_gain'] > 0:
                positive.append((record['fractional_gain'], physical))
    positive.sort(key=lambda item: (-item[0], item[1]))
    ordered_blocks = [physical for _, physical in positive]
    limit = min(cap, len(ordered_blocks))
    counts = sorted({count for count in (1, 2, 4, 8, limit) if 0 < count <= limit})
    prefixes = []
    best_mask, mask_name, mask_gain = zero, 'Q0', 0.
    for count in counts:
        selected = set(ordered_blocks[:count])
        proposal = make_role(m, n, k, world, chunk_rows, fixed_order,
                             [physical in selected for physical in range(blocks)])
        record = paired_measure(zero, proposal, evaluate, arm=SELECTIVE_ARM,
            pairs=cfg['training_pairs'], seed=seed + 3000000,
            phase='selective_prefix_training', name=f'prefix{count}')
        record['selected_physical_blocks'] = sorted(selected)
        prefixes.append(record)
        if record['fractional_gain'] > max(cfg['min_gain'], mask_gain):
            best_mask, mask_name, mask_gain = proposal, f'prefix{count}', record['fractional_gain']
    full = paired_measure(zero, all_quant, evaluate, arm=SELECTIVE_ARM,
        pairs=cfg['training_pairs'], seed=seed + 4000000,
        phase='selective_full_control', name='Qall')
    # Full quantization may win only if the declared budget allows it. The
    # control is always measured and numerically validated, even for budget=0.
    if cap == blocks and full['fractional_gain'] > max(cfg['min_gain'], mask_gain):
        best_mask, mask_name, mask_gain = all_quant, 'Qall', full['fractional_gain']
    mask_confirmation = None
    mask_accepted = False
    if mask_name != 'Q0':
        mask_confirmation = paired_measure(zero, best_mask, evaluate, arm=SELECTIVE_ARM,
            pairs=cfg['confirmation_pairs'], seed=seed + 5000000,
            phase='selective_confirmation', name=mask_name)
        mask_accepted = confirmed(mask_confirmation, cfg['min_gain'])
    if not mask_accepted:
        best_mask = zero
    final = make_role(m, n, k, world, chunk_rows, fixed_order, best_mask['selective_mask'])
    final['diagnostics'] = dict(
        method='same-codec-paired-complete-operator-selection-v2', settings=cfg,
        arrival=dict(selected_candidate=best_name if order_accepted else 'identity',
            proposed_candidate=best_name, accepted=order_accepted,
            identity_fallback=not order_accepted, training_candidates=order_trials,
            confirmation=order_confirmation,
            changed_slots=sum(a != b for a, b in zip(initial['arrival_order'], fixed_order))),
        selective=dict(selected_candidate=mask_name if mask_accepted else 'Q0',
            proposed_candidate=mask_name, accepted=mask_accepted, zero_fallback=not mask_accepted,
            budget_fraction=cfg['quant_budget'], budget_cap_blocks=cap,
            positive_singleton_blocks=ordered_blocks, singletons=singleton_trials,
            prefix_candidates=prefixes, full_control=full, confirmation=mask_confirmation),
        actual_selection=quantization_statistics(final, world),
        limitation='Role-shape operator calibration, not a guarantee of layer-specific or full-step benefit; confirmation does not reselect.')
    return validate_role_plan(final, m, n, k, world, chunk_rows)
