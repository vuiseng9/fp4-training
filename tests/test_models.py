import pytest
import torch
import torch.nn as nn
from models import TransformerBlock, TinyViT, TinyGPT, LINEAR_IMPL
from custom import CustomLinear

IMPL_TESTLIST = list(LINEAR_IMPL.keys())
NO_CPU_IMPL = ["te", "custom_aten", "cublaslt", "cublaslt_mxfp8", "cublaslt_nvfp4", "cublaslt_nvf4_fw_mxf8_bw"]

try:
    import transformer_engine.pytorch as te
except ImportError:
    Warning("transformer_engine.pytorch is not installed.")


class TestTransformerBlock:

    @pytest.mark.parametrize("linear_impl", IMPL_TESTLIST, ids=lambda x: f"Linear-{x}")
    def test_construction(self, linear_impl):
        txblk = TransformerBlock(E=64, F=128, H=4, impl=linear_impl)

        assert isinstance(txblk.attn.k_proj, LINEAR_IMPL[linear_impl])
        assert isinstance(txblk.ffn.down_proj, LINEAR_IMPL[linear_impl])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=lambda x: str(x).split(".")[-1])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("linear_impl", IMPL_TESTLIST, ids=lambda x: f"Linear-{x}")
    @pytest.mark.parametrize("is_causal", [False, True], ids=lambda x: f"(causal)" if x else "(bidir)")
    def test_forward(self, linear_impl, device, dtype, is_causal):
        emb_size = 64
        txblk = TransformerBlock(E=emb_size, F=emb_size*2, H=4, impl=linear_impl, is_causal=is_causal)
        txblk = txblk.to(device=device, dtype=dtype)

        # input sequences
        x = torch.randn(32, 10, emb_size, device=device, dtype=dtype)

        if linear_impl in NO_CPU_IMPL and device == "cpu":
            with pytest.raises((AssertionError, NotImplementedError, ValueError)):
                txblk(x)
        else:
            if is_causal is True:
                attn_mask = torch.tril(torch.ones((10, 10), dtype=torch.bool, device=device))
            else:
                attn_mask = None
            txblk(x, attn_mask=attn_mask)


class TestTinyViT:
    @pytest.mark.parametrize("linear_impl", IMPL_TESTLIST, ids=lambda x: f"linear_impl-{x}")
    def test_construction(self, linear_impl):
        vit = TinyViT(linear_impl=linear_impl)

        # check all the linear layers
        for n, m in vit.named_modules():
            if isinstance(m, nn.Linear):
                if linear_impl == "te" and "_proj" in n and not isinstance(m, te.Linear):
                    assert False, f"Module {n} is not an instance of te.Linear, got {type(m)}"
                elif linear_impl == "custom_py" and "_proj" in n and not isinstance(m, CustomLinear):
                    assert False, f"Module {n} is not an instance of CustomLinear, got {type(m)}"

class TestTinyGPT:
    @pytest.mark.parametrize("linear_impl", IMPL_TESTLIST, ids=lambda x: f"linear_impl-{x}")
    def test_construction(self, linear_impl):
        gpt = TinyGPT(linear_impl=linear_impl)

        # check all the linear layers
        for n, m in gpt.named_modules():
            if isinstance(m, nn.Linear):
                if linear_impl == "te" and "_proj" in n and not isinstance(m, te.Linear):
                    assert False, f"Module {n} is not an instance of te.Linear, got {type(m)}"
                elif linear_impl == "custom_py" and "_proj" in n and not isinstance(m, CustomLinear):
                    assert False, f"Module {n} is not an instance of CustomLinear, got {type(m)}"