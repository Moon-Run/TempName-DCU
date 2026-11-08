"""Sequence/tensor-parallel GPT training used by the DCU five-arm experiment.

PyTorch is imported only when building/running the model.  Configuration and
parameter-count validation therefore also run on a CPU host without PyTorch.
This is an explicit PyTorch training implementation, not a Megatron backend.
"""
from functools import lru_cache
import hashlib
import math
import time


DEFAULT_CONFIG = dict(layers=32, hidden=4096, ffn=16384, heads=32,
                      sequence=2048, micro_batch=1, accumulation_steps=4,
                      vocab=33792, chunk_rows=128)
ARMS = ("flux", "flux_taco", "flux_taco_interleaved",
        "flux_taco_interleaved_arrival", "flux_taco_interleaved_arrival_selective")


def validate_config(config, world_size=8):
    """Validate the actual model and all tensor/sequence partition sizes."""
    if not isinstance(config, dict):
        raise ValueError("model config must be a dictionary")
    result = dict(DEFAULT_CONFIG)
    result.update(config)
    for name in DEFAULT_CONFIG:
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    for name in ("hidden", "ffn", "heads", "vocab"):
        if result[name] % world_size:
            raise ValueError(f"{name} must be divisible by tensor-parallel size")
    if result["hidden"] % result["heads"]:
        raise ValueError("hidden must be divisible by heads")
    rows = result["sequence"] * result["micro_batch"]
    if rows % world_size or (rows // world_size) % result["chunk_rows"]:
        raise ValueError("sequence rows per rank must be divisible by chunk_rows")
    for key, default in (("learning_rate", 1e-4), ("weight_decay", 0.01),
                         ("max_grad_norm", 1.0), ("layer_norm_epsilon", 1e-5)):
        value = result.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < 0 or (key in ("learning_rate", "max_grad_norm", "layer_norm_epsilon") and value == 0):
            raise ValueError(f"invalid {key}")
        result[key] = float(value)
    return result


def parameter_counts(config, world_size=8):
    c = validate_config(config, world_size)
    h, f, v, s, layers = (c[k] for k in ("hidden", "ffn", "vocab", "sequence", "layers"))
    # QKV and FC1 biases are sharded; row biases, LN and positions replicated.
    logical = v*h + s*h + layers*(4*h*h + 2*h*f + 9*h + f) + 2*h
    replicated = s*h + layers*6*h + 2*h
    sharded = logical - replicated
    return dict(logical_parameters=logical, replicated_parameters=replicated,
                sharded_parameters=sharded, parameters_per_rank=sharded//world_size + replicated,
                tied_embedding_head=True)


@lru_cache(maxsize=1)
def _implementation():
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn

    def world(group):
        return dist.get_world_size(group) if dist.is_initialized() else 1

    def rank(group):
        return dist.get_rank(group) if dist.is_initialized() else 0

    def gather_rows(value, group):
        count = world(group)
        if count == 1:
            return value.contiguous()
        pieces = [torch.empty_like(value) for _ in range(count)]
        dist.all_gather(pieces, value.contiguous(), group=group)
        return torch.cat(pieces, dim=0)

    def scatter_sum(value, group):
        count = world(group)
        if count == 1:
            return value.contiguous()
        if value.shape[0] % count:
            raise ValueError("reduce-scatter rows do not divide the TP group")
        # Gloo's list reduce_scatter is unsupported in some PyTorch releases.
        # This branch is exclusively for the explicit CPU reference tests.
        if str(dist.get_backend(group)) == "gloo":
            copy = value.contiguous().clone()
            dist.all_reduce(copy, group=group)
            return copy.chunk(count, dim=0)[rank(group)].contiguous()
        result = torch.empty((value.shape[0]//count, *value.shape[1:]),
                             device=value.device, dtype=value.dtype)
        dist.reduce_scatter_tensor(result, value.contiguous(), group=group)
        return result

    class SumScatter(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, group):
            ctx.group = group
            return scatter_sum(value, group)

        @staticmethod
        def backward(ctx, gradient):
            return gather_rows(gradient, ctx.group), None

    class ColumnLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, local_input, weight, bias, group):
            full_input = gather_rows(local_input, group)
            ctx.save_for_backward(full_input, weight)
            ctx.group, ctx.has_bias = group, bias is not None
            output = F.linear(full_input, weight, bias)
            return output

        @staticmethod
        def backward(ctx, gradient):
            full_input, weight = ctx.saved_tensors
            gradient = gradient.contiguous()
            dinput = scatter_sum(gradient.matmul(weight), ctx.group)
            dweight = gradient.t().matmul(full_input)
            dbias = gradient.sum(0) if ctx.has_bias else None
            return dinput, dweight, dbias, None

    class RowLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, full_input, weight, operator, group):
            ctx.save_for_backward(full_input, weight)
            ctx.group = group
            # All five arms share exactly this autograd/STE backward.
            return operator.forward(full_input.contiguous(), weight.contiguous())

        @staticmethod
        def backward(ctx, gradient):
            full_input, weight = ctx.saved_tensors
            full_gradient = gather_rows(gradient, ctx.group)
            return (full_gradient.matmul(weight), full_gradient.t().matmul(full_input),
                    None, None)

    class VocabCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, targets, group):
            # FP32 softmax/CE is explicit even though GEMMs and activations are BF16.
            work = logits.float() if logits.dtype != torch.float64 else logits
            count, local_rank = world(group), rank(group)
            size = work.shape[-1]
            start = local_rank*size
            maximum = work.max(-1).values
            if count > 1:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            shifted = work - maximum.unsqueeze(-1)
            exponent = shifted.exp()
            denominator = exponent.sum(-1)
            if count > 1:
                dist.all_reduce(denominator, group=group)
            mask = (targets >= start) & (targets < start+size)
            indices = (targets-start).clamp(0, size-1)
            selected = work.gather(-1, indices.unsqueeze(-1)).squeeze(-1) * mask
            if count > 1:
                dist.all_reduce(selected, group=group)
            probability = exponent / denominator.unsqueeze(-1)
            ctx.save_for_backward(probability, indices, mask)
            ctx.input_dtype = logits.dtype
            return denominator.log() + maximum - selected

        @staticmethod
        def backward(ctx, gradient):
            probability, indices, mask = ctx.saved_tensors
            result = probability.clone()
            row = torch.arange(result.shape[0], device=result.device)
            result[row, indices] -= mask.to(result.dtype)
            result.mul_(gradient.unsqueeze(-1))
            return result.to(ctx.input_dtype), None, None

    class GPT(nn.Module):
        def __init__(self, config, arm, group, codec, device, dtype, seed, operator_factory):
            super().__init__()
            self.group, self.rank, self.world = group, rank(group), world(group)
            self.config = validate_config(config, self.world)
            self.arm, self.device, self.compute_dtype = arm, torch.device(device), dtype
            self.seed = seed
            c = self.config
            self.operators = []
            self.forward_calls = 0
            self.replicated_names = set()

            def param(name, shape, replicated=False, zero=False, one=False, std=0.02):
                tensor = torch.empty(shape, device=self.device, dtype=dtype)
                if zero:
                    tensor.zero_()
                elif one:
                    tensor.fill_(1)
                else:
                    # Name-derived seeds make arm construction order irrelevant.
                    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed((seed + salt + (0 if replicated else 1000003*self.rank)) % (2**63-1))
                    tensor.normal_(0, std, generator=generator)
                value = nn.Parameter(tensor)
                value._tp_replicated = replicated
                return value

            h, f = c["hidden"], c["ffn"]
            self.embedding = param("embedding", (c["vocab"]//self.world, h))
            self.position = param("position", (c["sequence"], h), replicated=True)
            self.layers = nn.ModuleList()
            for index in range(c["layers"]):
                prefix = f"layers.{index}"
                layer = nn.ParameterDict()
                for norm in ("ln1", "ln2"):
                    layer[norm+"_weight"] = param(prefix+norm+"w", (h,), replicated=True, one=True)
                    layer[norm+"_bias"] = param(prefix+norm+"b", (h,), replicated=True, zero=True)
                layer["qkv_weight"] = param(prefix+"qkvw", (3*h//self.world, h))
                layer["qkv_bias"] = param(prefix+"qkvb", (3*h//self.world,), zero=True)
                layer["attn_weight"] = param(prefix+"attnw", (h, h//self.world), std=0.02/math.sqrt(2*c["layers"]))
                layer["attn_bias"] = param(prefix+"attnb", (h,), replicated=True, zero=True)
                layer["fc1_weight"] = param(prefix+"fc1w", (f//self.world, h))
                layer["fc1_bias"] = param(prefix+"fc1b", (f//self.world,), zero=True)
                layer["fc2_weight"] = param(prefix+"fc2w", (h, f//self.world), std=0.02/math.sqrt(2*c["layers"]))
                layer["fc2_bias"] = param(prefix+"fc2b", (h,), replicated=True, zero=True)
                self.layers.append(layer)
                for k in (h//self.world, f//self.world):
                    self.operators.append(operator_factory(m=c["sequence"]*c["micro_batch"],
                        n=h, k=k, group=group, arm=arm, codec=codec, chunk_rows=c["chunk_rows"]))
            self.final_weight = param("final_ln_w", (h,), replicated=True, one=True)
            self.final_bias = param("final_ln_b", (h,), replicated=True, zero=True)

        def _norm(self, value, weight, bias):
            return F.layer_norm(value, (self.config["hidden"],), weight, bias,
                                self.config["layer_norm_epsilon"])

        def forward(self, tokens, targets):
            c = self.config
            m, h = c["sequence"]*c["micro_batch"], c["hidden"]
            if tokens.shape != (m,) or targets.shape != (m,):
                raise ValueError("tokens/targets must contain the full flattened microbatch")
            local_vocab = c["vocab"]//self.world
            start = local_vocab*self.rank
            belongs = (tokens >= start) & (tokens < start+local_vocab)
            ids = (tokens-start).clamp(0, local_vocab-1)
            contributions = F.embedding(ids, self.embedding) * belongs.unsqueeze(-1)
            value = SumScatter.apply(contributions, self.group)
            local_rows = m//self.world
            positions = torch.arange(self.rank*local_rows, (self.rank+1)*local_rows,
                                     device=tokens.device) % c["sequence"]
            value = value + F.embedding(positions, self.position)
            local_heads, head_dim = c["heads"]//self.world, h//c["heads"]
            for index, layer in enumerate(self.layers):
                norm = self._norm(value, layer["ln1_weight"], layer["ln1_bias"])
                qkv = ColumnLinear.apply(norm, layer["qkv_weight"], layer["qkv_bias"], self.group)
                q, k, v = qkv.reshape(c["micro_batch"], c["sequence"], 3, local_heads, head_dim).permute(2,0,3,1,4).unbind(0)
                attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                attention = attention.transpose(1,2).contiguous().reshape(m, h//self.world)
                value = value + RowLinear.apply(attention, layer["attn_weight"],
                                                self.operators[2*index], self.group) + layer["attn_bias"]
                norm = self._norm(value, layer["ln2_weight"], layer["ln2_bias"])
                intermediate = ColumnLinear.apply(norm, layer["fc1_weight"], layer["fc1_bias"], self.group)
                intermediate = F.gelu(intermediate, approximate="none")
                value = value + RowLinear.apply(intermediate, layer["fc2_weight"],
                                                self.operators[2*index+1], self.group) + layer["fc2_bias"]
            value = self._norm(value, self.final_weight, self.final_bias)
            logits = ColumnLinear.apply(value, self.embedding, None, self.group)
            self.forward_calls += 1
            return VocabCrossEntropy.apply(logits, targets, self.group).mean()

        def reduce_replicated_gradients(self, gradients=None):
            parameters = list(self.parameters())
            gradients = [p.grad for p in parameters] if gradients is None else gradients
            if len(gradients) != len(parameters):
                raise ValueError("Gradient list does not match model parameters")
            if self.world > 1:
                for parameter, gradient in zip(parameters, gradients):
                    if getattr(parameter, "_tp_replicated", False) and gradient is not None:
                        dist.all_reduce(gradient, group=self.group)

        def global_grad_norm(self, gradients=None):
            parameters = list(self.parameters())
            gradients = [p.grad for p in parameters] if gradients is None else gradients
            if len(gradients) != len(parameters):
                raise ValueError("Gradient list does not match model parameters")
            total = torch.zeros((), device=self.device, dtype=torch.float64)
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    raise RuntimeError("A trainable parameter has no gradient")
                if not getattr(parameter, "_tp_replicated", False) or self.rank == 0:
                    total += gradient.float().square().sum(dtype=torch.float64)
            if self.world > 1:
                dist.all_reduce(total, group=self.group)
            return total.sqrt()

    return dict(torch=torch, dist=dist, GPT=GPT, ColumnLinear=ColumnLinear,
                RowLinear=RowLinear, VocabCrossEntropy=VocabCrossEntropy,
                gather_rows=gather_rows, scatter_sum=scatter_sum)


def build_model(config, arm, group=None, codec=None, *, device="cuda", dtype=None,
                seed=1234, operator_factory=None, execution_plan=None):
    """Build the model; operator_factory is only for explicit CPU references."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    impl = _implementation()
    if operator_factory is None:
        from .runtime import FluxDCU
        operator_factory = FluxDCU
    if arm in ARMS[3:]:
        if execution_plan is None:
            raise ValueError("Arrival/selective model requires a frozen execution plan")
        factory = operator_factory
        from .planning import validate_role_plan
        count = impl['dist'].get_world_size(group) if impl['dist'].is_initialized() else 1
        c = validate_config(config, count)
        for role, width in (('attn', c['hidden']), ('mlp', c['ffn'])):
            validate_role_plan(execution_plan['roles'][role], c['sequence'] * c['micro_batch'],
                               c['hidden'], width // count, count, c['chunk_rows'])

        def operator_factory(**kwargs):
            role = 'attn' if kwargs['k'] == c['hidden'] // count else 'mlp'
            return factory(**kwargs, plan=execution_plan['roles'][role])
    torch = impl["torch"]
    return impl["GPT"](config, arm, group, codec, device,
                       torch.bfloat16 if dtype is None else dtype, seed, operator_factory)


def _tensor_digest(tensors):
    torch = _implementation()["torch"]
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_window(config, arm, group, codec, warmup_steps, timed_steps, seed=1234, token_seed=4321,
                 execution_plan=None):
    """Train one fresh model; all forward/backward/update work is timed.

    The parent process launches a new TP group for each arm/window.  This
    function never changes visible devices, creates a group or chooses a GPU.
    """
    impl = _implementation()
    torch, dist = impl["torch"], impl["dist"]
    if not dist.is_initialized() or str(dist.get_backend(group)) != "nccl":
        raise RuntimeError("DCU training requires an initialized PyTorch NCCL/RCCL group")
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("DCU training requires the target HIP/DTK PyTorch build")
    for name, value, lower in (("warmup_steps", warmup_steps, 1), ("timed_steps", timed_steps, 1)):
        if type(value) is not int or value < lower:
            raise ValueError(f"invalid {name}")
    count, local_rank = dist.get_world_size(group), dist.get_rank(group)
    c = validate_config(config, count)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = build_model(c, arm, group, codec, device=torch.device("cuda", torch.cuda.current_device()),
                        seed=seed, execution_plan=execution_plan)
    # Each real layer/role instance validates its full-shape partials, packets,
    # scales and output on its first warmup call.  No extra forward is inserted
    # and the checks cannot leak into the formal timing window.
    for operator in model.operators:
        operator.validate_next_call = True
    actual_count = sum(p.numel() for p in model.parameters())
    counts = parameter_counts(c, count)
    if actual_count != counts["parameters_per_rank"]:
        raise RuntimeError("Actual model parameter count differs from the frozen config")
    initial = _tensor_digest(model.named_parameters())
    generator = torch.Generator(device="cpu").manual_seed(token_seed)
    batches, token_tensors = [], []
    for micro in range(c["accumulation_steps"]):
        stream = torch.randint(0, c["vocab"], (c["micro_batch"], c["sequence"]+1), generator=generator)
        tokens, labels = stream[:,:-1].contiguous().flatten(), stream[:,1:].contiguous().flatten()
        token_tensors.extend(((f"tokens.{micro}", tokens), (f"labels.{micro}", labels)))
        batches.append((tokens.to(model.device), labels.to(model.device)))
    input_hash = _tensor_digest(token_tensors)
    parameters = list(model.parameters())
    masters = [torch.nn.Parameter(p.detach().float().clone()) for p in parameters]
    groups = [dict(params=[master for p,master in zip(parameters,masters) if p.ndim >= 2],
                   weight_decay=c["weight_decay"]),
              dict(params=[master for p,master in zip(parameters,masters) if p.ndim < 2],
                   weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=c["learning_rate"], betas=(0.9,0.95),
                                 eps=1e-8, weight_decay=c["weight_decay"], foreach=False)

    def step():
        optimizer.zero_grad(set_to_none=True)
        for parameter in parameters:
            parameter.grad = None
        loss_sum = torch.zeros((), device=model.device, dtype=torch.float32)
        for tokens, labels in batches:
            loss = model(tokens, labels)
            (loss / c["accumulation_steps"]).backward()
            loss_sum += loss.detach().float()/c["accumulation_steps"]
            # Accumulate microbatch gradients in FP32, instead of rounding
            # each accumulation back into the BF16 model parameter's .grad.
            for parameter, master in zip(parameters, masters):
                if parameter.grad is None:
                    raise RuntimeError("A trainable parameter has no microbatch gradient")
                if master.grad is None:
                    master.grad = parameter.grad.float()
                else:
                    master.grad.add_(parameter.grad)
                parameter.grad = None
        gradients = [master.grad for master in masters]
        model.reduce_replicated_gradients(gradients)
        norm = model.global_grad_norm(gradients)
        if not bool(torch.isfinite(norm).item()) or not bool(torch.isfinite(loss_sum).item()):
            raise RuntimeError("Nonfinite loss or global gradient norm")
        coefficient = (c["max_grad_norm"]/(norm+1e-6)).clamp(max=1.0).float()
        for master in masters:
            master.grad.mul_(coefficient)
        optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(parameters, masters):
                parameter.copy_(master)
        values = float(loss_sum.item()), float(norm.item())
        optimizer.zero_grad(set_to_none=True)
        for parameter in parameters:
            parameter.grad = None
        return values

    for _ in range(warmup_steps):
        step()
    # Warmup optimizer states and caches remain live throughout timing.
    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    calls_before = model.forward_calls
    operator_calls_before = [operator.calls for operator in model.operators]
    start = time.perf_counter()
    losses, norms = [], []
    for _ in range(timed_steps):
        loss, norm = step()
        losses.append(loss)
        norms.append(norm)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    final_hash = _tensor_digest(model.named_parameters())
    forwards = model.forward_calls-calls_before
    expected = timed_steps*c["accumulation_steps"]
    if forwards != expected:
        raise RuntimeError("Wrong formal model forward count")
    actual_calls = [operator.calls-before for operator,before in
                    zip(model.operators,operator_calls_before)]
    if len(actual_calls) != 2*c["layers"] or any(calls != expected for calls in actual_calls):
        raise RuntimeError("Actual FluxDCU calls disagree with formal projection coverage")
    output = dict(elapsed_seconds=elapsed, timed_steps=timed_steps, warmup_steps=warmup_steps,
        losses=losses, grad_norms=norms, tokens_per_step=c["sequence"]*c["micro_batch"]*c["accumulation_steps"],
        initialization_sha256=initial, input_sha256=input_hash, final_parameters_sha256=final_hash,
        actual_model_config=c, parameter_counts=counts, actual_parameters_per_rank=actual_count,
        dtype="bfloat16", optimizer="AdamW-fp32-master", deterministic_algorithms=True,
        gradient_policy="BF16 STE backward; FP32 microbatch accumulation; replicated gradients summed; replicated norm counted once",
        optimizer_weight_decay_policy="matrix parameters only; biases and LayerNorm vectors excluded",
        model_implementation="native-pytorch-sequence-tensor-parallel-gpt-v1",
        attention_implementation="torch.nn.functional.scaled_dot_product_attention-causal-no-dropout",
        fixed_synthetic_microbatches=True, rank=local_rank,
        operator_counters=dict(model_forwards=forwards, calls_per_projection=forwards,
                               attention_projection_calls=forwards*c["layers"],
                               mlp_projection_calls=forwards*c["layers"],
                               target_modules=2*c["layers"], actual_calls_per_instance=actual_calls,
                               actual_total_projection_calls=sum(actual_calls), fallback_calls=0),
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    # Read counters when available without inventing backend validation.
    states=[]
    for operator in model.operators:
        if hasattr(operator, "state") and callable(operator.state):
            states.append(operator.state())
    if states:
        output["operator_states"] = states
    return output
