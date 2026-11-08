"""Untimed finite-input and full-wire gates for the DCU operator.

Imports no torch until GPU/tensor helpers are called. The byte helper is a
stdlib-only validation oracle for tests and returned packet samples.
"""
import math
import struct


def _fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def validate_packet_bytes(packet, elements):
    """Validate exact SoA packet layout, finite FP8 coefficients, and scales."""
    packet = bytes(packet)
    if type(elements) is not int or elements <= 0 or elements % 128:
        raise ValueError('Packet element count must be a positive multiple of 128')
    groups = elements // 128
    if len(packet) != groups * 136:
        raise ValueError('Wrong TACO packet size')
    if any(value & 127 == 127 for value in packet[:elements]):
        raise ValueError('Nonfinite E4M3FN payload')
    quant = struct.unpack_from('<' + 'f' * groups, packet, elements)
    adaptive = struct.unpack_from('<' + 'f' * groups, packet, elements + groups * 4)
    if not all(math.isfinite(value) and _fp32(1e-12) <= value <= _fp32(1e6) for value in quant):
        raise ValueError('Invalid quantization scales')
    if not all(math.isfinite(value) and _fp32(1e-3) <= value <= _fp32(1e3) for value in adaptive):
        raise ValueError('Invalid adaptive scales')
    return dict(passed=True, elements=elements, groups=groups, bytes=len(packet))


def packet_tensor_valid(packet, elements):
    """Return a scalar device boolean without copying the full packet to CPU.

    CPU uint8 tensors are also accepted for unit tests. Structural violations
    raise ValueError; numeric violations return false for rank-coordinated gates.
    """
    import torch
    if type(elements) is not int or elements <= 0 or elements % 128:
        raise ValueError('Packet element count must be a positive multiple of 128')
    groups = elements // 128
    if (not isinstance(packet, torch.Tensor) or packet.dtype != torch.uint8 or
            not packet.is_contiguous() or packet.numel() != groups * 136 or packet.data_ptr() % 4):
        raise ValueError('Expected exact aligned contiguous uint8 TACO packet')
    flat = packet.view(-1)
    quant = flat[elements:elements + groups * 4].view(torch.float32)
    adaptive = flat[elements + groups * 4:].view(torch.float32)
    payload_valid = torch.bitwise_and(flat[:elements], 127).ne(127).all()
    quant_valid = (torch.isfinite(quant) & (quant >= _fp32(1e-12)) & (quant <= _fp32(1e6))).all()
    adaptive_valid = (torch.isfinite(adaptive) & (adaptive >= _fp32(1e-3)) & (adaptive <= _fp32(1e3))).all()
    return payload_valid & quant_valid & adaptive_valid


def validate_operator_buffers(op, input_, weight, output):
    """Synchronize and check every actual producer/receiver buffer, outside timing.

    Every rank performs one final MIN reduction before numeric/structural errors
    raise. A genuine device/communication fault may still abort via torchrun.
    Caller must invoke this helper on every rank at the same call boundary.
    """
    import torch
    import torch.distributed as dist
    torch.cuda.synchronize(op.device)
    good = torch.ones((), dtype=torch.bool, device=op.device)
    detail = []
    count = 0
    packet_count = 0
    elements = op.chunk_rows * op.n
    try:
        tensors = [('input', input_), ('weight', weight), ('output', output)]
        for index, block in enumerate(op.blocks):
            tensors.append((f'partial.{index}', block['partial']))
            if block['decoded'] is not None:
                tensors.append((f'decoded.{index}', block['decoded']))
            if not block['quantized'] and block['receive'].numel():
                tensors.append((f'receive.{index}', block['receive']))
        for name, tensor in tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16 or
                    not tensor.is_cuda or tensor.device.index != op.device):
                raise ValueError(f'Invalid BF16 tensor in validation: {name}')
            good = good & torch.isfinite(tensor).all()
            count += 1
        for index, block in enumerate(op.blocks):
            if block['quantized']:
                if block['send'].numel():
                    good = good & packet_tensor_valid(block['send'], elements)
                    packet_count += 1
                if block['receive'].numel():
                    received = block['receive'].view(op.world - 1, op.packet_bytes)
                    for source_slot in range(op.world - 1):
                        good = good & packet_tensor_valid(received[source_slot], elements)
                        packet_count += 1
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        good = torch.zeros((), dtype=torch.bool, device=op.device)
        detail.append(str(exc))
    flag = good.to(dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=op.group)
    if flag.item() != 1:
        suffix = ': ' + '; '.join(detail) if detail else ''
        raise RuntimeError('Full-shape finite-input/packet validation failed on at least one rank' + suffix)
    return dict(passed=True, finite_tensors_checked=count, packets_checked=packet_count,
                quantized_physical_blocks=sum(bool(block['quantized']) for block in op.blocks),
                bf16_physical_blocks=sum(not block['quantized'] for block in op.blocks),
                packet_groups_checked=packet_count * (elements // 128),
                payload_bytes_checked=packet_count * elements,
                scales='finite qscale in [FP32(1e-12),1e6], adaptive in [FP32(1e-3),1e3]',
                scope='untimed actual send/receive buffers; all H128 groups',
                rank_coordinated=True)
