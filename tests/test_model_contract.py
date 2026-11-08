"""CPU contract checks plus real Torch mathematical references when available."""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from flux_dcu.model import validate_config, parameter_counts

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _distributed_reference(rank, rendezvous):
    """Compare every TP parameter gradient with a separately evaluated dense GPT."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from flux_dcu.model import build_model, _implementation
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method="file://"+rendezvous)
    try:
        impl = _implementation()
        class DenseCollectiveOperator:
            def __init__(self, **kwargs): self.group=kwargs["group"]
            def forward(self, a, w):
                return impl["scatter_sum"](a.matmul(w.t()), self.group)
        config=dict(layers=1,hidden=8,ffn=16,heads=4,sequence=4,micro_batch=2,
                    accumulation_steps=1,vocab=16,chunk_rows=1)
        model=build_model(config,"flux",group=dist.group.WORLD,device="cpu",dtype=torch.float64,
                          seed=20261108,operator_factory=DenseCollectiveOperator)
        dense={}
        for name,p in model.named_parameters():
            values=[torch.empty_like(p) for _ in range(2)]
            dist.all_gather(values,p.detach())
            if getattr(p,"_tp_replicated",False):
                torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
                full=values[0]
            elif name.endswith("qkv_weight"):
                full=torch.cat([x.reshape(3,4,8) for x in values],dim=1).reshape(24,8)
            elif name.endswith("qkv_bias"):
                full=torch.cat([x.reshape(3,4) for x in values],dim=1).reshape(24)
            elif name.endswith(("attn_weight","fc2_weight")):
                full=torch.cat(values,dim=1)
            else:
                full=torch.cat(values,dim=0)
            dense[name]=full.detach().requires_grad_(True)
        tokens=torch.tensor([1,2,2,4,5,6,7,1]); labels=torch.tensor([2,2,4,8,6,7,1,9])
        actual=model(tokens,labels)
        # Independent unsharded path, no custom autograd or TP collectives.
        value=F.embedding(tokens,dense["embedding"])+F.embedding(torch.arange(8)%4,dense["position"])
        prefix="layers.0."
        norm=F.layer_norm(value,(8,),dense[prefix+"ln1_weight"],dense[prefix+"ln1_bias"],1e-5)
        qkv=F.linear(norm,dense[prefix+"qkv_weight"],dense[prefix+"qkv_bias"])
        q,k,v=qkv.reshape(2,4,3,4,2).permute(2,0,3,1,4).unbind(0)
        attn=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True)
        attn=attn.transpose(1,2).contiguous().reshape(8,8)
        value=value+F.linear(attn,dense[prefix+"attn_weight"],dense[prefix+"attn_bias"])
        norm=F.layer_norm(value,(8,),dense[prefix+"ln2_weight"],dense[prefix+"ln2_bias"],1e-5)
        value=value+F.linear(F.gelu(F.linear(norm,dense[prefix+"fc1_weight"],dense[prefix+"fc1_bias"])),
                             dense[prefix+"fc2_weight"],dense[prefix+"fc2_bias"])
        value=F.layer_norm(value,(8,),dense["final_weight"],dense["final_bias"],1e-5)
        expected=F.cross_entropy(F.linear(value,dense["embedding"]),labels)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        actual.backward(); expected.backward(); model.reduce_replicated_gradients()
        for name,p in model.named_parameters():
            gradient=dense[name].grad
            if getattr(p,"_tp_replicated",False):
                local=gradient
            elif name.endswith("qkv_weight"):
                local=gradient.reshape(3,8,8)[:,rank*4:(rank+1)*4].reshape(12,8)
            elif name.endswith("qkv_bias"):
                local=gradient.reshape(3,8)[:,rank*4:(rank+1)*4].reshape(12)
            elif name.endswith(("attn_weight","fc2_weight")):
                local=gradient.chunk(2,dim=1)[rank]
            else:
                local=gradient.chunk(2,dim=0)[rank]
            torch.testing.assert_close(p.grad,local,rtol=1e-8,atol=1e-9,msg=name)
        # Production norm squares FP32 values, so use that exact precision for
        # this independent dense norm while checking replicated deduplication.
        expected_norm=sum(p.grad.float().square().sum(dtype=torch.float64)
                          for p in dense.values()).sqrt()
        torch.testing.assert_close(model.global_grad_norm(),expected_norm,rtol=1e-7,atol=1e-9)
    finally:
        dist.destroy_process_group()


class ModelContract(unittest.TestCase):
    def test_default_is_full_model(self):
        c = validate_config({})
        self.assertEqual((c["layers"],c["hidden"],c["ffn"],c["heads"],c["sequence"]),
                         (32,4096,16384,32,2048))
        self.assertEqual(c["sequence"]*c["micro_batch"]*c["accumulation_steps"],8192)
        counts = parameter_counts(c)
        self.assertEqual(counts["logical_parameters"],6590963712)
        self.assertEqual((counts["parameters_per_rank"]-counts["replicated_parameters"])*8
                         + counts["replicated_parameters"],counts["logical_parameters"])

    def test_partition_validation(self):
        for changes in ({"heads":31},{"vocab":33791},{"hidden":4097},
                        {"sequence":2047},{"chunk_rows":192},{"layers":0},
                        {"micro_batch":True},{"learning_rate":math.nan}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_config(changes)


@unittest.skipUnless(HAS_TORCH,"PyTorch is not installed on this CPU development host")
class TorchReferences(unittest.TestCase):
    def setUp(self):
        import torch
        from flux_dcu.model import _implementation
        self.torch,self.impl=torch,_implementation()
        torch.manual_seed(2026)

    def test_vocab_ce_forward_and_gradient(self):
        torch=self.torch
        x=torch.randn(7,11,dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,10,3,2,2,5,8])
        reference=x.detach().clone().requires_grad_(True)
        actual=self.impl["VocabCrossEntropy"].apply(x,y,None).mean()
        expected=torch.nn.functional.cross_entropy(reference,y)
        actual.backward();expected.backward()
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(x.grad,reference.grad,rtol=1e-12,atol=1e-12)

    def test_column_and_row_autograd(self):
        torch=self.torch
        x=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
        w=torch.randn(6,4,dtype=torch.float64,requires_grad=True)
        b=torch.randn(6,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a,z,q:self.impl["ColumnLinear"].apply(a,z,q,None),(x,w,b)))
        class DenseOperator:
            def forward(self,a,z): return a.matmul(z.t())
        self.assertTrue(torch.autograd.gradcheck(
            lambda a,z:self.impl["RowLinear"].apply(a,z,DenseOperator(),None),(x,w)))

    def test_small_model_counts_and_loss_backward(self):
        torch=self.torch
        from flux_dcu.model import build_model
        class DenseOperator:
            def __init__(self,**kwargs): pass
            def forward(self,a,z): return a.matmul(z.t())
        config=dict(layers=2,hidden=8,ffn=16,heads=2,sequence=4,micro_batch=1,
                    accumulation_steps=1,vocab=16,chunk_rows=1)
        model=build_model(config,"flux",device="cpu",dtype=torch.float64,operator_factory=DenseOperator)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         parameter_counts(config,1)["parameters_per_rank"])
        tokens=torch.tensor([1,2,3,4]);labels=torch.tensor([2,3,4,5])
        loss=model(tokens,labels);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        # All targets changing must change actual CE, rather than a detached
        # scalar or a surrogate projection-only "training" workload.
        other=model(tokens,torch.tensor([6,7,8,9]))
        self.assertNotEqual(loss.item(),other.item())

    def test_two_rank_model_against_dense_forward_and_all_gradients(self):
        import torch.distributed as dist
        import torch.multiprocessing as mp
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("This PyTorch build has no CPU Gloo backend")
        with tempfile.TemporaryDirectory(prefix="flux-dcu-model-reference-") as folder:
            mp.spawn(_distributed_reference,args=(str(Path(folder)/"rendezvous"),),nprocs=2,join=True)


if __name__ == "__main__":
    unittest.main()
