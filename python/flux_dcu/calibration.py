"""Target-HIP calibration of complete operators; no profiler or BF16 tail labels."""
import json
import math
from pathlib import Path
import time

from .planning import SCHEMA, fit_role, role_digest, settings


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _reference(x, weight, plan, rank, world, group):
    import torch
    import torch.distributed as dist
    from .worker import independent_codec
    partial = torch.mm(x, weight.t()).contiguous()
    received = torch.empty_like(partial)
    dist.all_to_all_single(received.view(-1), partial.view(-1), group=group)
    rows, n = partial.shape[0] // world, partial.shape[1]
    sources = received.view(world, rows, n)
    plain = torch.zeros_like(sources[0])
    selected = torch.zeros_like(plain)
    per = rows // plan['chunk_rows']
    for source in range(world):
        plain = (plain + sources[source]).to(torch.bfloat16)
        contribution = sources[source].clone()
        if source != rank:
            for within in range(per):
                physical = rank * per + within
                if plan['selective_mask'][physical]:
                    start = within * plan['chunk_rows']
                    contribution[start:start + plan['chunk_rows']] = independent_codec(
                        contribution[start:start + plan['chunk_rows']])
        selected = (selected + contribution).to(torch.bfloat16)
    return plain, selected


def calibrate(config, group, codec, out):
    """Return one identical plan on all ranks and rank-local raw sample evidence.

    config requires model, calibration, seed, world_size. Every timing sample
    is reduced with MAX across ranks outside its timing interval. All native
    validation and framework references likewise execute outside the timer.
    """
    import torch
    import torch.distributed as dist
    from .runtime import FluxDCU
    from .worker import relative_error, require_all
    if not torch.version.hip:
        raise RuntimeError('Calibration requires target HIP/DTK; no CPU timing fallback')
    cfg = settings(config.get('calibration'))
    model = config['model']
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world != config['world_size']:
        raise ValueError('Calibration TP width differs from the frozen config')
    m, n, chunk = model['sequence'] * model['micro_batch'], model['hidden'], model['chunk_rows']
    samples = []
    destination = Path(out) / f'calibration-rank{rank}-samples.json'
    roles = {}
    for role_index, (role, k) in enumerate((('attn', model['hidden'] // world),
                                          ('mlp', model['ffn'] // world))):
        def evaluate(plan, seed, arm, label):
            generator = torch.Generator(device='cuda').manual_seed(seed + 173 * rank + 7919 * role_index)
            x = torch.randn((m, k), generator=generator, device='cuda', dtype=torch.bfloat16)
            weight = torch.randn((n, k), generator=generator, device='cuda', dtype=torch.bfloat16) / math.sqrt(k)
            op = FluxDCU(m, n, k, group, arm, codec, chunk, plan=plan)
            op.validate_next_call = True
            actual = op.forward(x, weight)
            # D always quantizes all remote sources, including when a final E
            # plan uses a sparse mask. Pilot D plans are nevertheless explicit.
            reference_plan = dict(plan)
            if arm == 'flux_taco_interleaved_arrival':
                reference_plan['selective_mask'] = [True] * len(plan['selective_mask'])
            plain, independent = _reference(x, weight, reference_plan, rank, world, group)
            errors = torch.tensor([relative_error(actual, plain), relative_error(actual, independent)],
                                  dtype=torch.float64, device='cuda')
            dist.all_reduce(errors, op=dist.ReduceOp.MAX, group=group)
            plain_error, codec_error = (float(v) for v in errors.cpu().tolist())
            require_all(math.isfinite(plain_error) and math.isfinite(codec_error)
                        and plain_error <= .05 and codec_error <= .01,
                        f'Calibration {role}/{label} failed independent numerical gate', group)
            del actual, plain, independent
            for _ in range(cfg['warmup']):
                op.forward(x, weight)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(cfg['iterations']):
                op.forward(x, weight)
            torch.cuda.synchronize()
            local = time.perf_counter() - started
            maximum = torch.tensor(local, dtype=torch.float64, device='cuda')
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            elapsed = float(maximum.item())
            sample = dict(role=role, label=label, seed=seed, arm=arm,
                          plan_digest=role_digest(plan), arrival_order=plan['arrival_order'],
                          selective_mask=plan['selective_mask'],
                          local_elapsed_seconds=local, max_rank_elapsed_seconds=elapsed,
                          iterations=cfg['iterations'], warmup=cfg['warmup'],
                          bf16_relative_l2=plain_error, codec_reference_relative_l2=codec_error,
                          correctness_passed=True)
            samples.append(sample)
            _write(destination, dict(schema='dcu-calibration-samples-v2', rank=rank, samples=samples))
            del op, x, weight
            return elapsed

        roles[role] = fit_role(m, n, k, world, chunk, evaluate, config=cfg,
                              seed=8000000 + 10000000 * role_index + int(config.get('seed', 1234)))
    bundle = dict(schema=SCHEMA, world_size=world, transport_lanes=2, roles=roles)
    digest = role_digest(bundle)
    gathered = [None] * world
    dist.all_gather_object(gathered, digest, group=group)
    if len(set(gathered)) != 1:
        raise RuntimeError('Calibration produced inconsistent plans across ranks')
    return dict(calibration_plan=bundle,
                calibration_results=dict(plan_sha256=digest, raw_samples=samples,
                    raw_samples_file=destination.name,
                    selection_timing='max-rank complete operator; references/profiling excluded',
                    training_and_confirmation_inputs_disjoint=True,
                    no_fullstep_claim=True))
