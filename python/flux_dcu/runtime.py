"""DCU BLAS/RCCL block pipeline with two physical transport lanes.

The five arms share one producer, two communication and two consumer streams.
Arrival arms use bounded per-lane ready consumption; this is not a global
dynamic scheduler or a claim of observed device overlap.
"""
import hashlib
import json
import threading

from .mapping import (ARRIVAL_ARMS, BACKEND, block_selection, schedule,
                      selection_statistics, split_sizes)

_CONTEXTS = {}
_CONTEXT_LOCK = threading.Lock()


def initialize_context(group):
    """Collectively create two RCCL groups in the same order on every rank.

    Call before constructing operators. Lazy construction remains available,
    but every rank must make that first call in the same process-group order.
    The current single-node experiment requires the parent group to cover all
    distributed ranks. Worker shutdown must destroy lane groups before WORLD.
    """
    import torch
    import torch.distributed as dist
    if not getattr(torch.version, 'hip', None):
        raise RuntimeError('FluxDCU requires HIP/DTK PyTorch; no CUDA/CPU fallback')
    if not dist.is_initialized() or str(dist.get_backend(group)) != 'nccl':
        raise RuntimeError('A HIP PyTorch NCCL-API/RCCL group is required')
    device = torch.cuda.current_device()
    key = (device, id(group))
    with _CONTEXT_LOCK:
        if key in _CONTEXTS:
            context = _CONTEXTS[key]
            if context['closed'] or context['failed']:
                raise RuntimeError('This FluxDCU context is closed or previously failed')
            return context
        size = dist.get_world_size(group)
        if size != dist.get_world_size():
            raise ValueError('This single-node context requires a parent group covering all ranks')
        if hasattr(dist, 'get_process_group_ranks'):
            ranks = list(dist.get_process_group_ranks(group))
        else:
            ranks = list(range(size))
        if ranks != list(range(size)):
            raise ValueError('Parent group must use ascending global rank order')
        # All ranks call new_group twice in exactly this order. All arms use
        # the same pair; collective submissions retain one common host order.
        groups = tuple(dist.new_group(ranks=ranks, backend='nccl') for _ in range(2))
        context = dict(group=group, groups=groups, ranks=ranks, device=device,
                       producer=torch.cuda.Stream(device=device),
                       communications=tuple(torch.cuda.Stream(device=device) for _ in range(2)),
                       consumers=tuple(torch.cuda.Stream(device=device) for _ in range(2)),
                       submission_lock=threading.Lock(), pending=[], last_done=None,
                       failed=False, closed=False)
        _CONTEXTS[key] = context
        return context


def shutdown_context(group):
    """Collective, ordered shutdown; invoke on every rank before WORLD teardown."""
    import torch
    import torch.distributed as dist
    key = (torch.cuda.current_device(), id(group))
    with _CONTEXT_LOCK:
        context = _CONTEXTS.get(key)
        if context is None:
            return
        if not context['submission_lock'].acquire(blocking=False):
            raise RuntimeError('Cannot shut down a context during operator submission')
        try:
            torch.cuda.synchronize(context['device'])
            for lane_group in context['groups']:
                dist.destroy_process_group(lane_group)
            context['pending'].clear()
            context['closed'] = True
            del _CONTEXTS[key]
        finally:
            context['submission_lock'].release()


class FluxDCU:
    def __init__(self, m, n, k, group, arm, codec, chunk_rows=128, plan=None,
                 diagnostic_delay_cycles=0):
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        self.group, self.codec, self.arm = group, codec, arm
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.device = torch.cuda.current_device()
        self.m, self.n, self.k, self.chunk_rows = m, n, k, chunk_rows
        if (type(diagnostic_delay_cycles) is not int or
                not 0 <= diagnostic_delay_cycles <= 100000000):
            raise ValueError('diagnostic_delay_cycles must be an integer in [0,100000000]')
        self.diagnostic_delay_cycles = diagnostic_delay_cycles
        self.last_profile_delay = None
        # Copy/freeze caller-owned plans so mutation cannot change wire format
        # after receiver allocations and process-group ordering are established.
        self.plan = json.loads(json.dumps(plan, allow_nan=False)) if plan is not None else None
        self.order = schedule(m, n, k, self.world, chunk_rows, arm, self.plan)
        self.selection_mask = block_selection(m, n, k, self.world, chunk_rows, arm, self.plan)
        self.consumer_policy = 'per_lane_ready' if arm in ARRIVAL_ARMS else 'ordered'
        self.quantized = any(self.selection_mask)
        self.mapping_sha256 = hashlib.sha256(json.dumps(self.order, sort_keys=True).encode()).hexdigest()
        if self.plan is not None:
            from .planning import role_digest
            self.plan_sha256 = role_digest(self.plan)
        else:
            self.plan_sha256 = None
        base = schedule(m, n, k, self.world, chunk_rows, 'flux_taco_interleaved')
        self.arrival_order_changed = [x['physical'] for x in self.order] != [x['physical'] for x in base] if self.plan else False
        self.selection = selection_statistics(m, n, self.world, chunk_rows, self.selection_mask)
        self.context = initialize_context(group)
        self.producer = self.context['producer']
        self.communications = self.context['communications']
        self.consumers = self.context['consumers']
        self.lane_groups = self.context['groups']
        # Legacy names refer only to lane 0. No execution relies on these aliases.
        self.communication, self.consumer = self.communications[0], self.consumers[0]
        self._lock = self.context['submission_lock']
        self._pending = self.context['pending']
        self.blocks, self.events = [], []
        self.calls = self.validation_calls = 0
        self.failed = self.profile_enabled = self.validate_next_call = False
        self._last_done = None
        elements = chunk_rows * n
        self.packet_bytes = elements // 128 * 136
        per_rank = m // self.world // chunk_rows
        for physical in range(len(self.order)):
            destination, within = divmod(physical, per_rank)
            local, selected = destination == self.rank, self.selection_mask[physical]
            partial = torch.empty((chunk_rows, n), dtype=torch.bfloat16, device=self.device)
            dtype = torch.uint8 if selected else torch.bfloat16
            count = self.packet_bytes if selected else elements
            send = (torch.empty(0, dtype=dtype, device=self.device) if local else
                    torch.empty(count, dtype=dtype, device=self.device) if selected else partial.view(-1))
            receive = torch.empty((self.world - 1) * count if local else 0,
                                  dtype=dtype, device=self.device)
            decoded = (torch.empty((self.world - 1, chunk_rows, n), dtype=torch.bfloat16,
                                   device=self.device) if local and selected else
                       receive.view(self.world - 1, chunk_rows, n) if local else None)
            sends, receives = split_sizes(self.world, self.rank, destination, count)
            self.blocks.append(dict(partial=partial, send=send, receive=receive, decoded=decoded,
                                    sends=sends, receives=receives, quantized=selected,
                                    physical=physical, destination=destination, within=within, lane=within % 2,
                                    ready=torch.cuda.Event(), received=torch.cuda.Event()))
        for block in self.blocks:
            for key in ('partial', 'send', 'receive', 'decoded'):
                value = block[key]
                if value is not None:
                    for stream in (self.producer, *self.communications, *self.consumers):
                        value.record_stream(stream)

    def _check(self, input_, weight):
        t = self.torch
        for label, value, shape in (('input', input_, (self.m, self.k)),
                                    ('weight', weight, (self.n, self.k))):
            if (tuple(value.shape) != shape or value.dtype != t.bfloat16 or
                    not value.is_cuda or value.device.index != self.device or not value.is_contiguous()):
                raise ValueError(f'{label} must be contiguous BF16 {shape} on device {self.device}')
        if self.failed or self.context['failed'] or self.context['closed']:
            raise RuntimeError('This operator/context previously failed or was closed and cannot be reused')

    def forward(self, input_, weight):
        import contextlib
        t = self.torch
        self._check(input_, weight)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent submissions to one FluxDCU process-group context are unsupported')
        try:
            caller = t.cuda.current_stream(self.device)
            self.producer.wait_stream(caller)
            if self.context['last_done'] is not None:
                self.producer.wait_event(self.context['last_done'])
            self._pending[:] = [(event, handles) for event, handles in self._pending if not event.query()]
            input_.record_stream(self.producer)
            weight.record_stream(self.producer)
            output = t.empty((self.m // self.world, self.n), dtype=t.bfloat16, device=self.device)
            for consumer in self.consumers:
                output.record_stream(consumer)
            handles, markers = [], []
            delayed = False
            if self.profile_enabled:
                self.last_profile_delay = None
            scope = (lambda name: t.autograd.profiler.record_function(name)) if self.profile_enabled else (lambda name: contextlib.nullcontext())
            for item in self.order:
                block = self.blocks[item['physical']]
                local, lane = item['destination'] == self.rank, block['lane']
                communication = self.communications[lane]
                with t.cuda.stream(self.producer), scope(f"flux_dcu/produce/l{lane}/p{item['physical']}"):
                    if self.profile_enabled:
                        begin = t.cuda.Event(enable_timing=True); begin.record()
                    t.mm(input_.narrow(0, item['row'], self.chunk_rows), weight.t(), out=block['partial'])
                    if block['quantized'] and not local:
                        self.codec.encode(block['partial'], block['send'])
                    if self.profile_enabled:
                        finish = t.cuda.Event(enable_timing=True); finish.record()
                        markers.append(('producer', item['physical'], lane, begin, finish))
                    block['ready'].record()
                with t.cuda.stream(communication), scope(f"flux_dcu/exchange/l{lane}/p{item['physical']}"):
                    communication.wait_event(block['ready'])
                    if self.profile_enabled:
                        begin = t.cuda.Event(enable_timing=True); begin.record()
                    if (self.profile_enabled and self.diagnostic_delay_cycles and
                            self.rank == 0 and lane == 0 and not delayed):
                        delay_begin = t.cuda.Event(enable_timing=True); delay_begin.record()
                        self.codec.delay(self.diagnostic_delay_cycles)
                        delay_end = t.cuda.Event(enable_timing=True); delay_end.record()
                        markers.append(('diagnostic_delay', item['physical'], lane, delay_begin, delay_end))
                        delayed = True
                        self.last_profile_delay = dict(physical=item['physical'], lane=lane,
                                                       rank=self.rank, cycles=self.diagnostic_delay_cycles)
                    work = self.dist.all_to_all_single(
                        block['receive'], block['send'], output_split_sizes=block['receives'],
                        input_split_sizes=block['sends'], group=self.lane_groups[lane], async_op=True)
                    # Host submit order is identical on every rank; complete
                    # Work.wait's stream-order contract before using another PG.
                    # Actual host blocking/concurrency depends on the HIP build.
                    work.wait()
                    handles.append(work)
                    if self.profile_enabled:
                        finish = t.cuda.Event(enable_timing=True); finish.record()
                        markers.append(('communication', item['physical'], lane, begin, finish))
                    block['received'].record()
                # Ordered controls retain all logical-slot waits on consumer0.
                # D/E skip nonlocal waits and only consume on the physical lane,
                # so a different lane's delayed receive need not block this lane.
                if self.consumer_policy == 'ordered' or local:
                    consumer = self.consumers[0 if self.consumer_policy == 'ordered' else lane]
                    with t.cuda.stream(consumer), scope(f"flux_dcu/consume/l{lane}/p{item['physical']}"):
                        consumer.wait_event(block['received'])
                        if local:
                            if self.profile_enabled:
                                begin = t.cuda.Event(enable_timing=True); begin.record()
                            if block['quantized']:
                                packets = block['receive'].view(self.world - 1, self.packet_bytes)
                                for source_slot in range(self.world - 1):
                                    self.codec.decode(packets[source_slot], block['decoded'][source_slot])
                            target = output.narrow(0, item['within'] * self.chunk_rows, self.chunk_rows)
                            self.codec.reduce(block['partial'], block['decoded'], target, self.rank, self.world)
                            if self.profile_enabled:
                                finish = t.cuda.Event(enable_timing=True); finish.record()
                                markers.append(('consumer', item['physical'], lane, begin, finish))
            # Final join includes BOTH transport lanes, even on ranks which did
            # not consume a block on a lane. Protect all send/receive scratch and
            # output visibility before a caller or subsequent producer advances.
            for stream in (*self.communications, *self.consumers):
                self.producer.wait_stream(stream)
            done = t.cuda.Event()
            done.record(self.producer)
            caller.wait_event(done)
            self._last_done = self.context['last_done'] = done
            self._pending.append((done, handles))
            self.events = markers
            self.calls += 1
            if self.validate_next_call:
                from .validation import validate_operator_buffers
                validate_operator_buffers(self, input_, weight, output)
                self.validate_next_call = False
                self.validation_calls += 1
            return output
        except BaseException:
            self.failed = self.context['failed'] = True
            raise
        finally:
            self._lock.release()

    def profile_call(self, input_, weight):
        """Separate diagnostic events; intervals are not kernel-overlap proof."""
        t = self.torch
        t.cuda.synchronize()
        origin = t.cuda.Event(enable_timing=True)
        origin.record()
        self.profile_enabled = True
        try:
            result = self.forward(input_, weight)
            t.cuda.synchronize()
            rows = [dict(stage=stage, physical=physical, lane=lane,
                         start_ms=origin.elapsed_time(start), end_ms=origin.elapsed_time(end))
                    for stage, physical, lane, start, end in self.events]
        finally:
            self.profile_enabled = False
        overlap = sum(max(0., min(a['end_ms'], b['end_ms']) - max(a['start_ms'], b['start_ms']))
                      for a in rows for b in rows
                      if a['stage'] == 'producer' and b['stage'] == 'communication')
        return result, dict(events=rows, producer_communication_interval_overlap_ms=overlap,
                            transport_lanes=2, consumer_policy=self.consumer_policy,
                            diagnostic_delay=self.last_profile_delay,
                            diagnostic_limitation='Artificial skew tests mechanism availability, not normal-workload performance',
                            interval_limitation='Intervals can include waits and overlapping intervals can double-count time; not proof of RCCL kernel overlap.')

    def state(self):
        total = sum(value.numel() * value.element_size() for block in self.blocks
                    for key in ('partial', 'send', 'receive', 'decoded')
                    if (value := block[key]) is not None and
                    not (key == 'send' and not block['quantized'] and value.numel()) and
                    not (key == 'decoded' and not block['quantized']))
        quantization = ('selective_remote_h128_e4m3' if self.arm.endswith('_selective') else
                        'all_remote_h128_e4m3' if self.arm != 'flux' else 'none')
        return dict(backend=BACKEND, arm=self.arm, rank=self.rank, calls=self.calls,
                    shape=[self.m, self.n, self.k], chunk_rows=self.chunk_rows,
                    production_schedule=self.order, mapping_sha256=self.mapping_sha256,
                    gemms_per_call=len(self.order), collectives_per_call=len(self.order),
                    quantization=quantization, selection_mask=list(self.selection_mask),
                    selection_granularity='physical_receiver_block_all_remote_sources',
                    selected_remote_blocks_by_source=self.selection['selected_remote_blocks_by_source'],
                    remote_wire_bytes_by_source=self.selection['remote_wire_bytes_by_source'],
                    remote_wire_bytes_per_source=self.selection['remote_wire_bytes_by_source'][self.rank],
                    actual_remote_wire_bytes_per_rank=self.selection['remote_wire_bytes_by_source'][self.rank],
                    plan_sha256=self.plan_sha256, arrival_order_changed=self.arrival_order_changed,
                    reduction_order=list(range(self.world)), rounding='bf16_after_each_source',
                    scratch_bytes=total, original_cuda_flux=False, cta_level_swizzle=False,
                    shared_streams_per_group=5, transport_lanes=2,
                    lane_assignment='physical_within_destination_mod_2',
                    consumer_policy=self.consumer_policy, validation_calls=self.validation_calls,
                    diagnostic_delay_cycles=self.diagnostic_delay_cycles,
                    diagnostic_delay_in_normal_forward=False,
                    scheduling_granularity='blas_macroblock', overlap_claim='requires_device_trace',
                    communicator_limitation='Two RCCL communicators express independent lanes; no concurrency or speedup is guaranteed')
