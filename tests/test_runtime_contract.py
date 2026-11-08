"""CPU event-DAG tests for runtime lifetime and stream dependencies.

The fake backend records enqueue dependencies and storage identities. It does
not execute GPU arithmetic, model RCCL scheduling, or claim actual overlap.
"""
import contextlib
import importlib.util
import math
from pathlib import Path
import sys
import struct
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from flux_dcu.runtime import FluxDCU, initialize_context, shutdown_context
from flux_dcu import reference
from flux_dcu.planning import make_role
from flux_dcu.validation import validate_packet_bytes, packet_tensor_valid


class Device:
    type = "cuda"
    index = 0


class Tensor:
    next_storage = 0
    def __init__(self, shape, dtype, storage=None, offset=0):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device, self.offset = dtype, Device(), offset
        self.is_cuda, self.recorded = True, []
        if storage is None:
            Tensor.next_storage += 1
            storage = Tensor.next_storage
        self.storage = storage
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 1 if self.dtype == "uint8" else 2
    def is_contiguous(self): return True
    def record_stream(self, stream): self.recorded.append(stream)
    def view(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.numel() // math.prod(x for x in shape if x != -1)
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype, self.storage, self.offset)
    def narrow(self, dim, start, length):
        assert dim == 0 and start >= 0 and start + length <= self.shape[0]
        return Tensor((length, *self.shape[1:]), self.dtype, self.storage,
                      self.offset + start * math.prod(self.shape[1:]))
    def t(self): return Tensor(self.shape[::-1], self.dtype, self.storage, self.offset)
    def __getitem__(self, index):
        assert isinstance(index, int) and 0 <= index < self.shape[0]
        return Tensor(self.shape[1:], self.dtype, self.storage,
                      self.offset + index * math.prod(self.shape[1:]))


class Event:
    def __init__(self, cuda): self.cuda, self.node, self.completed = cuda, None, False
    def record(self, stream=None): self.node = (stream or self.cuda.current).last
    def query(self): return self.completed


class Stream:
    def __init__(self, cuda): self.cuda, self.last = cuda, None
    def wait_event(self, event): self.cuda.submit("wait_event", self, extra=[event.node])
    def wait_stream(self, stream): self.cuda.submit("wait_stream", self, extra=[stream.last])


class FakeCuda:
    def __init__(self):
        self.nodes = []
        self.current = Stream(self)
    def current_device(self): return 0
    def synchronize(self, device=None): pass
    def current_stream(self, device=None): return self.current
    def Stream(self, device=None): return Stream(self)
    def Event(self, **kwargs): return Event(self)
    @contextlib.contextmanager
    def stream(self, value):
        old, self.current = self.current, value
        try: yield
        finally: self.current = old
    def submit(self, kind, stream=None, extra=(), **metadata):
        stream = stream or self.current
        dependencies = [item for item in [stream.last, *extra] if item is not None]
        ancestors = set(dependencies)
        for item in dependencies: ancestors.update(self.nodes[item]["ancestors"])
        node = dict(kind=kind, ancestors=ancestors, stream=stream, **metadata)
        stream.last = len(self.nodes)
        self.nodes.append(node)
        return stream.last


class FakeCodec:
    def __init__(self, cuda): self.cuda = cuda
    def encode(self, source, packet): self.cuda.submit("encode", source=source, output=packet)
    def decode(self, packet, output): self.cuda.submit("decode", source=packet, output=output)
    def delay(self, cycles, stream=None): self.cuda.submit("diagnostic_delay", cycles=cycles)
    def reduce(self, local, remote, output, rank, world):
        self.cuda.submit("reduce", local=local, remote=remote, output=output, rank=rank, world=world)


class FakeBackend:
    def __init__(self, rank=0):
        self.cuda = cuda = FakeCuda()
        self.works = []
        self.group = object()
        self.groups, self.destroyed = [], []
        self.networks = []
        self.torch = types.ModuleType("torch")
        self.dist = types.ModuleType("torch.distributed")
        self.torch.cuda, self.torch.version = cuda, types.SimpleNamespace(hip="fake-CPU-test")
        self.torch.bfloat16, self.torch.uint8 = "bf16", "uint8"
        self.torch.autograd = types.SimpleNamespace(profiler=types.SimpleNamespace(
            record_function=lambda name: contextlib.nullcontext()))
        self.torch.empty = lambda shape, *, dtype, device: Tensor(shape, dtype)
        def mm(a, b, *, out):
            assert a.shape[1] == b.shape[0] and out.shape == (a.shape[0], b.shape[1])
            cuda.submit("gemm", source=a, weight=b, output=out)
        self.torch.mm = mm
        self.dist.is_initialized = lambda: True
        self.dist.get_backend = lambda group: "nccl"
        self.dist.get_rank = lambda group=None: rank
        self.dist.get_world_size = lambda group=None: 2
        self.dist.get_process_group_ranks = lambda group: [0, 1]
        def new_group(*, ranks, backend):
            assert ranks == [0, 1] and backend == 'nccl'
            group = types.SimpleNamespace(lane=len(self.groups), ranks=ranks)
            self.groups.append(group); self.networks.append(Stream(cuda))
            return group
        self.dist.new_group = new_group
        self.dist.destroy_process_group = lambda group: self.destroyed.append(group)
        def exchange(receive, send, output_split_sizes, input_split_sizes, group, async_op):
            assert async_op and sum(output_split_sizes) == receive.numel() and sum(input_split_sizes) == send.numel()
            launch = cuda.submit("exchange", self.networks[group.lane], extra=[cuda.current.last],
                                 send=send, receive=receive, group=group,
                                 sends=list(input_split_sizes), receives=list(output_split_sizes))
            work = types.SimpleNamespace(node=launch, waits=0)
            def wait():
                work.waits += 1
                cuda.submit("work_wait", extra=[launch])
            work.wait = wait
            self.works.append(work)
            return work
        self.dist.all_to_all_single = exchange
        self.torch.distributed = self.dist
        self.codec = FakeCodec(cuda)
    def modules(self): return patch.dict(sys.modules, {"torch": self.torch, "torch.distributed": self.dist})
    def operator(self, arm, plan=None, diagnostic_delay_cycles=0):
        return FluxDCU(8, 128, 4, self.group, arm, self.codec, chunk_rows=2,
                       plan=plan, diagnostic_delay_cycles=diagnostic_delay_cycles)
    def inputs(self): return Tensor((8, 4), "bf16"), Tensor((128, 4), "bf16")


class RuntimeContract(unittest.TestCase):
    def test_production_order_changes_actual_blas_input_rows(self):
        for arm, rows in (("flux", [0, 2, 4, 6]), ("flux_taco", [0, 2, 4, 6]),
                          ("flux_taco_interleaved", [0, 4, 2, 6])):
            backend = FakeBackend()
            with backend.modules():
                op = backend.operator(arm)
                x, w = backend.inputs(); op.forward(x, w)
            actual = [n["source"].offset // 4 for n in backend.cuda.nodes if n["kind"] == "gemm"]
            self.assertEqual(actual, rows)

    def test_consumer_dependency_and_returned_output_fence(self):
        backend = FakeBackend(rank=1)
        with backend.modules():
            op = backend.operator("flux_taco_interleaved")
            x, w = backend.inputs(); output = op.forward(x, w)
            done = op._last_done.node
        exchanges = [i for i,n in enumerate(backend.cuda.nodes) if n["kind"] == "exchange"]
        reductions = [i for i,n in enumerate(backend.cuda.nodes) if n["kind"] == "reduce"]
        self.assertEqual(len(reductions), 2)
        self.assertTrue(all(i in backend.cuda.nodes[done]["ancestors"] for i in exchanges))
        returned = backend.cuda.current.last
        self.assertIn(done, backend.cuda.nodes[returned]["ancestors"])
        for index in reductions:
            node = backend.cuda.nodes[index]
            # Quantized receive and decoded storage differ. The reduction still
            # has an exchange and decode dependency in its transitive closure.
            self.assertTrue(any(i in node["ancestors"] for i in exchanges))
            self.assertTrue(any(n["kind"] == "decode" and i in node["ancestors"]
                                for i,n in enumerate(backend.cuda.nodes)))
            self.assertEqual(node["output"].storage, output.storage)
        self.assertTrue(all(work.waits == 1 for work in backend.works))
        self.assertEqual(len(op._pending[0][1]), 4)

    def test_reuse_waits_previous_consumption_and_outputs_are_independent(self):
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator("flux_taco")
            x, w = backend.inputs(); first = op.forward(x, w)
            first_done = op._last_done.node
            before = len(backend.cuda.nodes)
            second = op.forward(x, w)
        second_gemms = [n for n in backend.cuda.nodes[before:] if n["kind"] == "gemm"]
        self.assertNotEqual(first.storage, second.storage)
        self.assertTrue(all(first_done in node["ancestors"] for node in second_gemms))
        self.assertEqual(len(op._pending), 2)
        self.assertIn(op.producer, x.recorded)
        self.assertIn(op.producer, w.recorded)

    def test_submission_failure_poisoning_prevents_scratch_reuse(self):
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator("flux_taco")
            x,w = backend.inputs()
            def fail(*args, **kwargs): raise RuntimeError("fake transport failure")
            backend.dist.all_to_all_single = fail
            with self.assertRaisesRegex(RuntimeError, "fake transport"):
                op.forward(x,w)
            self.assertTrue(op.failed)
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                op.forward(x,w)

    def test_context_groups_are_shared_and_destroyed_in_creation_order(self):
        backend = FakeBackend()
        with backend.modules():
            context = initialize_context(backend.group)
            first = backend.operator('flux')
            second = backend.operator('flux_taco')
            self.assertIs(first.context, context)
            self.assertIs(second.producer, first.producer)
            self.assertEqual(len(backend.groups), 2)
            self.assertEqual(len({id(x) for x in (context['producer'], *context['communications'], *context['consumers'])}), 5)
            shutdown_context(backend.group)
            self.assertEqual(backend.destroyed, list(context['groups']))
            self.assertTrue(context['closed'])
            x,w = backend.inputs()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                first.forward(x,w)

    def test_ready_lane_bypasses_other_lane_but_final_join_covers_everything(self):
        plan = make_role(8,128,4,2,2)
        for arm in ('flux_taco_interleaved', 'flux_taco_interleaved_arrival'):
            backend = FakeBackend()
            with backend.modules():
                op = backend.operator(arm, None if arm == 'flux_taco_interleaved' else plan)
                x,w = backend.inputs();op.forward(x,w)
            nodes = backend.cuda.nodes
            first_lane0 = next(i for i,n in enumerate(nodes) if n['kind']=='exchange' and n['group'].lane==0)
            lane1_reduce = next(n for n in nodes if n['kind']=='reduce' and n['output'].offset==256)
            if arm == 'flux_taco_interleaved':
                self.assertIn(first_lane0, lane1_reduce['ancestors'])
            else:
                self.assertNotIn(first_lane0, lane1_reduce['ancestors'])
                self.assertIs(lane1_reduce['stream'], op.consumers[1])
                self.assertEqual(sum(n['kind']=='wait_event' and n['stream'] in op.consumers for n in nodes), 2)
            done = nodes[op._last_done.node]
            self.assertTrue(all(i in done['ancestors'] for i,n in enumerate(nodes)
                                if n['kind'] in ('exchange','reduce')))

    def test_physical_lane_is_unchanged_by_arrival_permutation(self):
        plan = make_role(8,128,4,2,2,order=[1,3,0,2])
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator('flux_taco_interleaved_arrival',plan)
            x,w = backend.inputs();op.forward(x,w)
        self.assertEqual([b['lane'] for b in op.blocks],[0,1,0,1])
        self.assertEqual([x['lane'] for x in op.order],[1,1,0,0])
        self.assertEqual([n['group'].lane for n in backend.cuda.nodes if n['kind']=='exchange'],[1,1,0,0])
        self.assertTrue(op.state()['arrival_order_changed'])

    def test_selective_closed_block_formats_q0_partial_full(self):
        for mask in ([False]*4,[True,False,False,True],[True]*4):
            traces=[]
            for rank in range(2):
                backend=FakeBackend(rank)
                plan=make_role(8,128,4,2,2,mask=mask)
                with backend.modules():
                    op=backend.operator('flux_taco_interleaved_arrival_selective',plan)
                    x,w=backend.inputs();op.forward(x,w)
                for physical,block in enumerate(op.blocks):
                    self.assertEqual(block['send'].dtype,'uint8' if mask[physical] else 'bf16')
                    self.assertEqual(block['receive'].dtype,block['send'].dtype)
                    self.assertEqual(block['quantized'],mask[physical])
                    if not mask[physical] and block['send'].numel():
                        self.assertEqual(block['send'].storage,block['partial'].storage)
                    if not mask[physical] and block['decoded'] is not None:
                        self.assertEqual(block['decoded'].storage,block['receive'].storage)
                state=op.state()
                selected=sum(mask[p] for p in range(4) if p//2!=rank)
                self.assertEqual(state['selected_remote_blocks_by_source'][rank],selected)
                self.assertEqual(state['actual_remote_wire_bytes_per_rank'],selected*272+(2-selected)*512)
                self.assertEqual(sum(n['kind']=='encode' for n in backend.cuda.nodes),selected)
                traces.append([n for n in backend.cuda.nodes if n['kind']=='exchange'])
            self.assertEqual([n['group'].lane for n in traces[0]],[n['group'].lane for n in traces[1]])
            for zero,one in zip(*traces):
                self.assertEqual(zero['sends'][1],one['receives'][0])
                self.assertEqual(one['sends'][0],zero['receives'][1])

    def test_diagnostic_delay_is_profile_only_rank0_lane0_once(self):
        plan=make_role(8,128,4,2,2)
        for rank in (0,1):
            backend=FakeBackend(rank)
            with backend.modules():
                op=backend.operator('flux_taco_interleaved_arrival',plan,diagnostic_delay_cycles=12345)
                x,w=backend.inputs();op.forward(x,w)
                self.assertFalse(any(n['kind']=='diagnostic_delay' for n in backend.cuda.nodes))
                op.profile_enabled=True
                op.forward(x,w)
            delays=[n for n in backend.cuda.nodes if n['kind']=='diagnostic_delay']
            self.assertEqual(len(delays),1 if rank==0 else 0)
            if rank==0:
                self.assertIs(delays[0]['stream'],op.communications[0])
                self.assertEqual(op.last_profile_delay,dict(physical=0,lane=0,rank=0,cycles=12345))
            self.assertFalse(op.state()['diagnostic_delay_in_normal_forward'])


class PacketValidationContract(unittest.TestCase):
    def test_full_soa_packet_and_fp32_minimum_scales(self):
        packet = reference.encode([0.0] * 256)
        result = validate_packet_bytes(packet, 256)
        self.assertTrue(result['passed'])
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['bytes'], 272)

    def test_nonfinite_payload_and_corrupt_scales_any_group(self):
        original = bytearray(reference.encode([1.0] * 256))
        for index, byte in ((0, 127), (255, 255)):
            damaged = original.copy(); damaged[index] = byte
            with self.assertRaisesRegex(ValueError, 'payload'):
                validate_packet_bytes(damaged, 256)
        for index, value in ((256, math.nan), (260, math.inf), (260, 0.0),
                             (264, -1.0), (268, 1001.0)):
            damaged = original.copy(); struct.pack_into('<f', damaged, index, value)
            with self.assertRaisesRegex(ValueError, 'scales'):
                validate_packet_bytes(damaged, 256)

    def test_wrong_size_and_group_count(self):
        for packet, elements in ((bytes(135),128),(bytes(272),128),(bytes(136),129)):
            with self.assertRaises(ValueError):
                validate_packet_bytes(packet,elements)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,
                         'PyTorch unavailable; stdlib wire gates still tested')
    def test_tensor_packet_checker_on_cpu(self):
        import torch
        raw = reference.encode([0.0] * 256)
        packet = torch.tensor(list(raw),dtype=torch.uint8)
        self.assertTrue(packet_tensor_valid(packet,256).item())
        packet[255] = 255
        self.assertFalse(packet_tensor_valid(packet,256).item())
        packet = torch.tensor(list(raw),dtype=torch.uint8)
        packet[260:264] = torch.tensor(list(struct.pack('<f', math.nan)),dtype=torch.uint8)
        self.assertFalse(packet_tensor_valid(packet,256).item())


if __name__ == "__main__":
    unittest.main()
