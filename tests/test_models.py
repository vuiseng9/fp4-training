import pytest
import torch
import torch.nn as nn
from models import TransformerBlock, TinyViT, LINEAR_IMPL
from custom import CustomLinear

IMPL_TESTLIST = list(LINEAR_IMPL.keys())

try:
    import transformer_engine.pytorch as te
except ImportError:
    Warning("transformer_engine.pytorch is not installed.")


class TestTransformerBlock:

    @pytest.mark.parametrize("linear_impl", IMPL_TESTLIST, ids=lambda x: f"Linear-{x}")
    def test_construction(self, linear_impl):
        txblk = TransformerBlock(E=64, F=128, H=4, impl=linear_impl)

        match linear_impl:
            case "torch":
                assert isinstance(txblk.attn.k_proj, nn.Linear)
                assert isinstance(txblk.ffn.down_proj, nn.Linear)
            
            case "te":
                assert isinstance(txblk.attn.k_proj, te.Linear)
                assert isinstance(txblk.ffn.down_proj, te.Linear)

            case "custom_py":
                assert isinstance(txblk.attn.k_proj, CustomLinear)
                assert isinstance(txblk.ffn.down_proj, CustomLinear)

            case _:
                assert False, f"should never end up here, pls debug"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=lambda x: str(x).split(".")[-1])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("linear_impl", IMPL_TESTLIST, ids=lambda x: f"Linear-{x}")
    def test_forward(self, linear_impl, device, dtype):
        emb_size = 64
        txblk = TransformerBlock(E=emb_size, F=emb_size*2, H=4, impl=linear_impl)
        txblk = txblk.to(device=device, dtype=dtype)

        # input sequences
        x = torch.randn(32, 10, emb_size, device=device, dtype=dtype)
        try:
            y = txblk(x)
        except AssertionError as e:
            if linear_impl == "te" and device == "cpu":
                assert True, "te.Linear is not for CPU"
            else:
                assert False, f"Unexpected error: {e}"


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
