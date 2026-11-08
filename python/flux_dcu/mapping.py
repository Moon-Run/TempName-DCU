"""Rank-independent production schedules; no device imports or side effects.

The interleaved mapping follows Entwine v1 section 3.1 equation (1).
This backend schedules BLAS macroblocks, not CUTLASS thread blocks.
"""

ARMS = ("flux", "flux_taco", "flux_taco_interleaved",
        "flux_taco_interleaved_arrival", "flux_taco_interleaved_arrival_selective")
ARRIVAL_ARMS = ARMS[3:]
BACKEND = "dcu-blas-rccl-block-lanes-v2"


def validate_shape(m, n, k, world, chunk_rows):
    values = (m, n, k, world, chunk_rows)
    if any(type(x) is not int or x <= 0 for x in values):
        raise ValueError("M, N, K, world and chunk_rows must be positive integers")
    if world < 2 or m % (world * chunk_rows):
        raise ValueError("M must be divisible by world_size * chunk_rows; world >= 2")
    if n % 128:
        raise ValueError("N must be a multiple of the H128 codec group")


def schedule(m, n, k, world, chunk_rows, arm, plan=None):
    validate_shape(m, n, k, world, chunk_rows)
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    per_rank = m // world // chunk_rows
    if arm in ARRIVAL_ARMS:
        validate_runtime_plan(plan, m, n, k, world, chunk_rows)
        order = plan['arrival_order']
    elif plan is not None:
        raise ValueError("The first three arms must not accept an arrival/selection plan")
    else:
        order = None
    result = []
    for slot in range(world * per_rank):
        if order is not None:
            destination, within = divmod(order[slot], per_rank)
        elif arm == "flux_taco_interleaved":
            destination, within = slot % world, slot // world
        else:
            destination, within = divmod(slot, per_rank)
        physical = destination * per_rank + within
        result.append(dict(slot=slot, destination=destination, within=within,
                           physical=physical, row=physical * chunk_rows,
                           rows=chunk_rows, lane=within % 2))
    return result


def validate_runtime_plan(plan, m, n, k, world, chunk_rows):
    """Check the scheduling/wire contract before delegating evidence validation."""
    validate_shape(m, n, k, world, chunk_rows)
    if not isinstance(plan, dict):
        raise ValueError('Arrival/selective arms require an explicit role plan')
    if plan.get('shape') != [m, n, k] or plan.get('chunk_rows') != chunk_rows:
        raise ValueError('Role plan shape or chunk_rows differs from the operator')
    count = m // chunk_rows
    per_rank = count // world
    order, mask = plan.get('arrival_order'), plan.get('selective_mask')
    if (not isinstance(order, list) or len(order) != count or
            any(type(value) is not int for value in order) or sorted(order) != list(range(count))):
        raise ValueError('arrival_order must be a complete physical-block permutation')
    if any(physical // per_rank != slot % world for slot, physical in enumerate(order)):
        raise ValueError('Arrival order must preserve each interleaved destination slot')
    if not isinstance(mask, list) or len(mask) != count or any(type(value) is not bool for value in mask):
        raise ValueError('selective_mask must have one bool per physical block')
    if not isinstance(plan.get('diagnostics'), dict):
        raise ValueError('Role plan diagnostics must be a dictionary')
    # The independent planner validates provenance/plan-specific fields. The
    # lazy import avoids making basic mapping tests depend on calibration code.
    from .planning import validate_role_plan
    validate_role_plan(plan, shape=[m, n, k], world_size=world, chunk_rows=chunk_rows)
    return plan


def block_selection(m, n, k, world, chunk_rows, arm, plan=None):
    """Receiver-block closure: every remote source uses the same wire format."""
    schedule(m, n, k, world, chunk_rows, arm, plan)
    if arm == ARMS[-1]:
        return list(plan['selective_mask'])
    return [arm != 'flux'] * (m // chunk_rows)


def selection_statistics(m, n, world, chunk_rows, mask):
    validate_shape(m, n, 1, world, chunk_rows)
    count = m // chunk_rows
    if (not isinstance(mask, list) or len(mask) != count or
            any(type(value) is not bool for value in mask)):
        raise ValueError('Invalid closed-block selection dimensions/mask')
    per_rank = count // world
    bf16_bytes = chunk_rows * n * 2
    compressed_bytes = chunk_rows * n // 128 * 136
    selected, remote_bytes = [], []
    for source in range(world):
        remote = [index for index in range(count) if index // per_rank != source]
        chosen = sum(mask[index] for index in remote)
        selected.append(chosen)
        remote_bytes.append(chosen * compressed_bytes + (len(remote) - chosen) * bf16_bytes)
    return dict(selected_remote_blocks_by_source=selected,
                remote_blocks_per_source=count - per_rank,
                remote_wire_bytes_by_source=remote_bytes,
                selected_physical_blocks=sum(mask), physical_blocks=count,
                bf16_bytes_per_block=bf16_bytes, quantized_bytes_per_block=compressed_bytes)


def wire_bytes(m, n, world, quantized):
    elements = (world - 1) * (m // world) * n
    return elements // 128 * 136 if quantized else elements * 2


def split_sizes(world, rank, destination, elements):
    """Sparse All-to-All: all remote sources send one block to destination.

    Self data never goes through RCCL. Receive slots are source-rank ordered.
    """
    if not 0 <= rank < world or not 0 <= destination < world or elements <= 0:
        raise ValueError("Invalid rank/destination/count")
    send = [0] * world
    receive = [0] * world
    if rank != destination:
        send[destination] = elements
    else:
        receive = [elements if source != rank else 0 for source in range(world)]
    return send, receive
