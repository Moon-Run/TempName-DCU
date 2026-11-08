"""One torchrun window. GPU checks precede all recorded performance results."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback

from flux_dcu.mapping import ARMS, BACKEND


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor_sha(tensor):
    value = tensor.detach().contiguous().view(-1).view(__import__('torch').uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def require_all(ok, message, group):
    import torch
    import torch.distributed as dist
    flag = torch.tensor(int(bool(ok)), dtype=torch.int32, device='cuda')
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if flag.item() != 1:
        raise RuntimeError(message + ' (failed on at least one rank)')


def identity(rank):
    import torch
    p = torch.cuda.get_device_properties(rank)
    result = dict(host=socket.gethostname(), local_rank=rank, device_name=p.name,
                  device_count=torch.cuda.device_count(), torch=torch.__version__,
                  hip=torch.version.hip, python=platform.python_version(),
                  backend='nccl-api-on-hip', compute_units=p.multi_processor_count,
                  total_memory=p.total_memory,
                  gcn_arch=getattr(p, 'gcnArchName', None),
                  uuid=str(getattr(p, 'uuid', 'unavailable')),
                  pci_bus_id=str(getattr(p, 'pci_bus_id', 'unavailable')),
                  visibility={k: os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')},
                  communication_environment={k: v for k, v in sorted(os.environ.items())
                     if k.startswith(('NCCL_', 'RCCL_', 'TORCH_NCCL_', 'HSA_'))})
    # Do not infer a CUDA architecture or warp size from HIP device properties.
    return result


def native_checks(codec, group):
    import torch
    from flux_dcu import reference
    generator = torch.Generator(device='cuda').manual_seed(9871)
    cases = [torch.zeros(256, device='cuda', dtype=torch.bfloat16),
             torch.randn(256, generator=generator, device='cuda').to(torch.bfloat16)]
    outlier = cases[1].clone(); outlier[0] = 128.; outlier[128] = -96.; cases.append(outlier)
    errors = []
    for x in cases:
        packet = torch.empty(x.numel() // 128 * 136, dtype=torch.uint8, device='cuda')
        actual = torch.empty_like(x)
        codec.encode(x, packet)
        codec.decode(packet, actual)
        torch.cuda.synchronize()
        packed = bytes(packet.cpu().tolist())
        # Decode actual bytes using independent FP64 CPU math; independently
        # encode the same BF16 input as a separate end-to-end codec oracle.
        byte_reference = torch.tensor(reference.decode(packed), device='cuda')
        independent = torch.tensor(reference.decode(reference.encode(x.float().cpu().tolist())), device='cuda')
        byte_error = relative_error(actual.float(), byte_reference)
        algorithm_error = relative_error(actual.float(), independent)
        good = bool(torch.isfinite(actual).all()) and byte_error <= .01 and algorithm_error <= .02
        require_all(good, f'HIP codec / independent CPU oracle failed: {byte_error}, {algorithm_error}', group)
        errors.append(dict(actual_packet_reference_relative_l2=byte_error,
                           independent_codec_relative_l2=algorithm_error))
    return errors


def relative_error(actual, expected):
    import torch
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return float('inf')
    difference = torch.linalg.vector_norm(actual.float() - expected.float()).item()
    norm = torch.linalg.vector_norm(expected.float()).item()
    return difference / max(norm, 1e-12)


def preflight(config, rank, world, group, codec):
    import torch
    import torch.distributed as dist
    from flux_dcu.mapping import split_sizes
    from flux_dcu.runtime import initialize_context
    context = initialize_context(group)
    # Exercise real BF16 BLAS instead of trusting dtype/device declarations.
    a = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8).to(torch.bfloat16) / 32
    b = a.t().contiguous()
    product = torch.mm(a, b)
    ref = (a.double().cpu() @ b.double().cpu()).to(torch.bfloat16).float().to('cuda')
    require_all(product.dtype == torch.bfloat16 and relative_error(product, ref) < .01,
                'BF16 BLAS capability check failed', group)
    probes = []
    for lane, lane_group in enumerate(context['groups']):
        for dtype in (torch.bfloat16, torch.uint8):
            for destination in range(world):
                sends, receives = split_sizes(world, rank, destination, 128)
                tx = torch.full((sum(sends),), rank + 1, device='cuda', dtype=dtype)
                rx = torch.empty(sum(receives), device='cuda', dtype=dtype)
                dist.all_to_all_single(rx, tx, receives, sends, group=lane_group)
                if rank == destination:
                    expected = torch.cat([torch.full((128,), s + 1, device='cuda', dtype=dtype)
                                          for s in range(world) if s != rank])
                    correct = torch.equal(rx, expected)
                else:
                    correct = rx.numel() == 0
                require_all(correct, 'RCCL lane/zero-split All-to-All failed; no transport fallback', group)
                probes.append(dict(lane=lane, dtype=str(dtype), destination=destination, passed=True))
    values = torch.full((world * 4,), rank + 1, dtype=torch.bfloat16, device='cuda')
    reduced = torch.empty(4, dtype=torch.bfloat16, device='cuda')
    dist.reduce_scatter_tensor(reduced, values, group=group)
    require_all(torch.equal(reduced, torch.full_like(reduced, world * (world + 1) // 2)),
                'BF16 reduce_scatter_tensor failed', group)
    peer = []
    for other in range(world):
        try:
            peer.append(bool(torch.cuda.can_device_access_peer(rank, other)))
        except RuntimeError:
            peer.append(None)
    smi = None
    if rank == 0:
        records = {}
        for flag in ('--showproductname', '--showbus', '--showtopo', '--showdriverversion'):
            try:
                probe = subprocess.run(['hy-smi', flag], capture_output=True, text=True, timeout=20)
                records[flag] = dict(returncode=probe.returncode, stdout=probe.stdout, stderr=probe.stderr)
            except (OSError, subprocess.TimeoutExpired) as exc:
                records[flag] = dict(error=str(exc))
        smi = records
    return dict(capabilities=dict(bf16_gemm=True, bf16_reduce_scatter=True,
                    sparse_alltoall_bf16=True, sparse_alltoall_uint8=True,
                    peer_access=peer, direct_peer_required=False, transport_lanes=2),
                collective_probes=probes, codec_checks=native_checks(codec, group), hy_smi=smi)


def hadamard(values):
    """Independent high-level tensor butterfly; no native codec implementation."""
    x = values.float().reshape(-1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pair = x.reshape(-1, 128 // (2 * stride), 2, stride)
        x = __import__('torch').stack((pair[:, :, 0] + pair[:, :, 1],
                                      pair[:, :, 0] - pair[:, :, 1]), dim=2).reshape(-1, 128)
    return x / math.sqrt(128)


def independent_codec(values):
    """FP32 framework reference with nearest-code lookup, no float8 support needed."""
    import torch
    from flux_dcu.reference import e4m3_decode
    original_shape = values.shape
    x = values.float().reshape(-1, 128)
    adaptive = (448. / torch.sqrt((x * x).mean(1, keepdim=True) + 1e-6)).clamp(1e-3, 1e3)
    z = hadamard(x * adaptive)
    scale = (z.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.).clamp(1e-12, 1e6)
    scaled = (z / scale).clamp(-448, 448)
    levels = torch.tensor([e4m3_decode(code) for code in range(127)], device=values.device)
    absolute = scaled.abs().contiguous()
    upper = torch.bucketize(absolute, levels).clamp(0, 126)
    lower = (upper - 1).clamp_min(0)
    lo_distance, hi_distance = absolute - levels[lower], levels[upper] - absolute
    choose_upper = (hi_distance < lo_distance) | ((hi_distance == lo_distance) & ((upper % 2) == 0))
    quantized = levels[torch.where(choose_upper, upper, lower)] * scaled.sign()
    result = hadamard(quantized * scale) / adaptive
    return result.reshape(original_shape).to(torch.bfloat16)


def reduce_reference_sources(sources, rank, quantized, selection_mask=None, chunk_rows=None):
    """Independent fixed-order reference, including closed physical-block masks."""
    import torch
    world, local_rows, _ = sources.shape
    if selection_mask is not None:
        if chunk_rows is None or local_rows % chunk_rows:
            raise ValueError('Independent mask reference requires an exact chunk size')
        per = local_rows // chunk_rows
        if len(selection_mask) != world * per or any(type(x) is not bool for x in selection_mask):
            raise ValueError('Independent mask reference requires a physical boolean block mask')
    bf16 = torch.zeros_like(sources[0])
    independent = torch.zeros_like(bf16)
    for source in range(world):
        bf16 = (bf16 + sources[source]).to(torch.bfloat16)
        contribution = sources[source]
        if quantized and source != rank:
            if selection_mask is None:
                contribution = independent_codec(contribution)
            else:
                contribution = contribution.clone()
                for within in range(per):
                    if selection_mask[rank * per + within]:
                        block = contribution.narrow(0, within * chunk_rows, chunk_rows)
                        block.copy_(independent_codec(block))
        independent = (independent + contribution).to(torch.bfloat16)
    return bf16, independent


def projection_reference(input_, weight, rank, world, group, quantized,
                         selection_mask=None, chunk_rows=None):
    import torch
    import torch.distributed as dist
    # Full GEMM deliberately differs from the producer's chunk calls.
    partial = torch.mm(input_, weight.t()).contiguous()
    sources = torch.empty_like(partial)
    dist.all_to_all_single(sources.view(-1), partial.view(-1), group=group)
    sources = sources.view(world, partial.shape[0] // world, partial.shape[1])
    return reduce_reference_sources(sources, rank, quantized, selection_mask, chunk_rows)


def arrival_diagnostic(config, arm, role_plan, input_, weight, group, codec):
    """Artificial skew proves possible bypass, never ordinary-run speedup.

    Use base interleaving so delayed lane0 is submitted before lane1. The
    selected physical mask is preserved. This independent operator is excluded
    from calibration, the formal operator counters, and all timed samples.
    """
    from copy import deepcopy
    import torch
    from flux_dcu.runtime import FluxDCU
    from flux_dcu.planning import base_order
    world = config['world_size']
    m, k = input_.shape
    n = weight.shape[0]
    chunk = config['model']['chunk_rows']
    plan = deepcopy(role_plan)
    plan['arrival_order'] = base_order(m, n, k, world, chunk)
    cycles = 50000000
    probe = FluxDCU(m, n, k, group, arm, codec, chunk, plan=plan,
                    diagnostic_delay_cycles=cycles)
    probe.validate_next_call = True
    normal = probe.forward(input_, weight)
    delayed, diagnostics = probe.profile_call(input_, weight)
    require_all(torch.equal(normal, delayed), 'Artificial lane skew changed output', group)
    slots = {physical: slot for slot, physical in enumerate(plan['arrival_order'])}
    pending_physical = plan['arrival_order'][0]
    communication = next(row for row in diagnostics['events']
                         if row['stage'] == 'communication' and row['physical'] == pending_physical)
    overtakes = [row['physical'] for row in diagnostics['events']
                 if row['stage'] == 'consumer' and row['lane'] != 0
                 and slots[row['physical']] > slots[pending_physical]
                 and row['end_ms'] < communication['end_ms']]
    return dict(passed=True, numerical_bitwise_equal=True, artificial=True,
                delayed_rank=0, delayed_lane=0, delayed_physical=pending_physical,
                delay_cycles=cycles, later_blocks_completed_before_pending=overtakes,
                ready_bypass_observed=bool(overtakes),
                evidence=diagnostics, ordinary_speedup_claim=False,
                limitation='Artificial skew, receiver-local event ordering; no cross-rank clock comparison or normal workload speedup claim.')


def operators(config, arm, rank, world, group, codec, out):
    import torch
    import torch.distributed as dist
    from flux_dcu.runtime import FluxDCU
    model, timing = config['model'], config['timing']
    m, n = model['sequence'] * model['micro_batch'], model['hidden']
    results = []
    states = []
    input_hashes = []
    for role, k in (('attn', model['hidden'] // world), ('mlp', model['ffn'] // world)):
        generator = torch.Generator(device='cuda').manual_seed(config['seed'] + 173 * rank + k)
        x = torch.randn((m, k), generator=generator, device='cuda', dtype=torch.bfloat16)
        w = torch.randn((n, k), generator=generator, device='cuda', dtype=torch.bfloat16) / math.sqrt(k)
        input_hashes.extend((tensor_sha(x), tensor_sha(w)))
        role_plan = config['execution_plan']['roles'][role] if arm in ARMS[3:] else None
        mask = role_plan['selective_mask'] if arm == ARMS[4] else None
        op = FluxDCU(m, n, k, group, arm, codec, model['chunk_rows'], plan=role_plan)
        op.validate_next_call = True
        output = op.forward(x, w)
        bf16, expected = projection_reference(x, w, rank, world, group, arm != 'flux', mask, model['chunk_rows'])
        plain_error, codec_error = relative_error(output, bf16), relative_error(output, expected)
        require_all(plain_error <= (.05 if arm != 'flux' else .01) and codec_error <= .01,
                    f'{role} full-shape reference failed: BF16={plain_error}, codec={codec_error}', group)
        retained = output.clone()
        # Change real inputs, then reuse the same scratch; old returned storage
        # must remain valid and the next invocation must not consume stale data.
        x_changed = (x * .75 + .125).to(torch.bfloat16).contiguous()
        op.validate_next_call = True
        changed = op.forward(x_changed, w)
        _, expected_changed = projection_reference(x_changed, w, rank, world, group,
                                                   arm != 'flux', mask, model['chunk_rows'])
        reuse_error = relative_error(changed, expected_changed)
        require_all(torch.equal(output, retained) and reuse_error <= .01,
                    f'{role} output lifetime / changed-input reuse failed', group)
        bitwise_order = None
        if arm != 'flux':
            other_plan = None
            if arm == ARMS[4]:
                from copy import deepcopy
                from flux_dcu.planning import base_order
                other = arm
                other_plan = deepcopy(role_plan)
                other_plan['arrival_order'] = base_order(m, n, k, world, model['chunk_rows'])
            else:
                other = 'flux_taco' if arm == 'flux_taco_interleaved' else 'flux_taco_interleaved'
            control = FluxDCU(m, n, k, group, other, codec, model['chunk_rows'], plan=other_plan)
            other_output = control.forward(x, w)
            bitwise_order = torch.equal(other_output, retained)
            require_all(bitwise_order, f'{role} order-only output changed; cannot report fair scheduling ablation', group)
            torch.cuda.synchronize()
            del other_output, control
        ready_diagnostic = (arrival_diagnostic(config, arm, role_plan, x, w, group, codec)
                            if arm in ARMS[3:] else None)
        del bf16, expected, expected_changed, changed, retained, output, x_changed
        diagnostic = None
        if config.get('profile'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                _, diagnostic = op.profile_call(x, w)
            trace = out / f'rank{rank}-{role}-trace.json'
            profiler.export_chrome_trace(str(trace))
            diagnostic['trace'] = trace.name
            diagnostic['trace_sha256'] = hashlib.sha256(trace.read_bytes()).hexdigest()
            diagnostic['actual_kernel_overlap_validated'] = False
        for _ in range(timing['operator_warmup']):
            op.forward(x, w)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(timing['operator_iterations']):
            op.forward(x, w)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        # Useful independent control: full BLAS without RS, timed separately.
        full = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        for _ in range(5):
            torch.mm(x, w.t(), out=full)
        torch.cuda.synchronize(); dist.barrier(group=group); torch.cuda.synchronize()
        start_gemm = time.perf_counter()
        for _ in range(timing['operator_iterations']):
            torch.mm(x, w.t(), out=full)
        torch.cuda.synchronize()
        full_gemm_elapsed = time.perf_counter() - start_gemm
        state = op.state()
        states.append(state)
        results.append(dict(role=role, elapsed_seconds=elapsed,
                            timed_iterations=timing['operator_iterations'], correctness_passed=True,
                            bf16_relative_l2=plain_error, codec_reference_relative_l2=codec_error,
                            changed_input_relative_l2=reuse_error, order_bitwise_equal=bitwise_order,
                            full_gemm_only_seconds=full_gemm_elapsed, profile=diagnostic,
                            arrival_diagnostic=ready_diagnostic,
                            execution=state))
        torch.cuda.synchronize()
        del op, x, w, full
    return dict(operator_results=results, operators=states,
                input_sha256=hashlib.sha256(''.join(input_hashes).encode()).hexdigest(),
                initialization_sha256=hashlib.sha256(''.join(input_hashes).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('preflight', 'calibrate', 'operator', 'train'), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    row = dict(rank=rank, world_size=world, stage=args.stage, arm=args.arm,
               run_id=config['run_id'], completed=False, passed=False,
               execution_plan_sha256=config.get('execution_plan_sha256'),
               backend=BACKEND, original_cuda_flux=False, cta_level_swizzle=False)
    try:
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        from flux_dcu.codec import NativeCodec
        if not torch.version.hip:
            raise RuntimeError('Target must provide HIP/DTK PyTorch. No CPU/NVIDIA fallback is implemented.')
        if world != config['world_size'] or int(os.environ['LOCAL_RANK']) != rank:
            raise ValueError('Single-node torchrun rank/config mismatch')
        if torch.cuda.device_count() < world:
            raise RuntimeError(f'Require {world} visible DCUs, found {torch.cuda.device_count()}')
        torch.cuda.set_device(rank)
        torch.set_num_threads(1)
        # All arms use the same deterministic math. Unsupported deterministic
        # target operators fail rather than weaken the scheduling comparison.
        torch.use_deterministic_algorithms(True)
        dist.init_process_group('nccl', timeout=timedelta(seconds=600))
        group = dist.group.WORLD
        from flux_dcu.runtime import initialize_context, shutdown_context
        initialize_context(group)
        if args.stage in ('operator', 'train'):
            from flux_dcu.planning import validate_role_plan
            bundle = config.get('execution_plan')
            expected_hash = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(',', ':'),
                                                       allow_nan=False).encode()).hexdigest()
            if not bundle or expected_hash != config.get('execution_plan_sha256'):
                raise ValueError('Missing or changed frozen calibration plan')
            if bundle.get('world_size') != world or bundle.get('transport_lanes') != 2:
                raise ValueError('Plan topology mismatch')
            model = config['model']
            for role, width in (('attn', model['hidden']), ('mlp', model['ffn'])):
                validate_role_plan(bundle['roles'][role], model['sequence'] * model['micro_batch'],
                                   model['hidden'], width // world, world, model['chunk_rows'])
        row['environment_identity'] = identity(rank)
        arch = getattr(torch.cuda.get_device_properties(rank), 'gcnArchName', '').split(':')[0]
        if arch and not os.environ.get('FLUX_DCU_HIP_ARCH'):
            os.environ['FLUX_DCU_HIP_ARCH'] = arch
        if not os.environ.get('FLUX_DCU_HIP_ARCH'):
            raise RuntimeError('HIP architecture not exposed by this torch build; set FLUX_DCU_HIP_ARCH to the verified target gfx architecture')
        if arch and os.environ['FLUX_DCU_HIP_ARCH'] != arch:
            raise RuntimeError('FLUX_DCU_HIP_ARCH disagrees with the actual device architecture')
        codec = NativeCodec(hipcc=config.get('hipcc'), build_dir=Path(config['build_dir']), verbose=rank == 0)
        row['codec_build'] = codec.manifest
        if args.stage == 'preflight':
            row.update(preflight(config, rank, world, group, codec))
        elif args.stage == 'calibrate':
            from flux_dcu.calibration import calibrate
            row.update(calibrate(config, group, codec, args.out))
        elif args.stage == 'operator':
            row['codec_checks'] = native_checks(codec, group)
            row.update(operators(config, args.arm, rank, world, group, codec, args.out))
        else:
            from flux_dcu.model import train_window
            row.update(train_window(config['model'], args.arm, group, codec,
                                    config['timing']['warmup_steps'], config['timing']['timed_steps'],
                                    config['seed'], config['token_seed'],
                                    execution_plan=config['execution_plan']))
        torch.cuda.synchronize()
        row.update(completed=True, passed=True)
        write(args.out / f'rank{rank}.json', row)
        dist.barrier()
        shutdown_context(group)
        dist.destroy_process_group()
    except BaseException as exc:
        row.update(completed=False, passed=False, error=str(exc), traceback=traceback.format_exc())
        write(args.out / f'rank{rank}.json', row)
        raise


if __name__ == '__main__':
    main()
