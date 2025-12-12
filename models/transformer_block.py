###
# locally implemented layers
import os
import torch
from torch import nn
import torch.nn.functional as F
from custom import (
    CustomLinear, 
    AddmmLinear, 
    CublasltLinear, 
    CublasltMxfp8Linear,
    CublasltNvfp4Linear,
    CublasltFwdNvfp4BwdMxfp8Linear,
)

LINEAR_IMPL = {
    "torch": nn.Linear,
    "custom_py": CustomLinear,
    "custom_aten": AddmmLinear,
    "cublaslt": CublasltLinear,
    "cublaslt_mxfp8": CublasltMxfp8Linear,
    "cublaslt_nvfp4": CublasltNvfp4Linear, 
    "cublaslt_nvf4_fw_mxf8_bw": CublasltFwdNvfp4BwdMxfp8Linear,
}

try:
    import transformer_engine.pytorch as te
    LINEAR_IMPL["te"] = te.Linear
except ImportError:
    Warning("transformer_engine.pytorch is not installed.")
    te = None

REF_IMPL = ["torch", "custom_py", "custom_aten", "cublaslt", "cublaslt_mxfp8", "cublaslt_nvfp4", "cublaslt_nvf4_fw_mxf8_bw"]

class TransformerBlock(nn.Module):
    def __init__(self, E, F, H, dropout=0.1, impl="torch", is_causal=False):
        super().__init__()
        self.E = E
        self.F = F
        self.H = H
        self.impl = impl
        self.is_causal = is_causal

        self.attn = AttentionBlock(E, H, dropout, impl=impl, is_causal=is_causal)

        self.ffn = nn.ModuleDict({
            "preln": nn.LayerNorm(E),
            "up_proj": LINEAR_IMPL[impl](E, F),
            "act": nn.GELU(),
            "act_dropout": nn.Dropout(dropout),
            "down_proj": LINEAR_IMPL[impl](F, E),
            "dropout": nn.Dropout(dropout),
        })

    def forward(self, x, attn_mask=None):
        # x in (B, L, E)
        # Note Image shape transformation
        # B, C(E), H, W
        # → B, (H*W), C(E)
        if attn_mask is not None and self.is_causal is False:
            raise ValueError("Non-causal attention should not have an attention mask.")
        
        if self.is_causal is True and attn_mask is None:
            raise ValueError("Causal attention requires an attention mask.")
        
        hidden = self.attn(x, attn_mask=attn_mask)
        residual = hidden

        for _, layer in self.ffn.items():
            hidden = layer(hidden)

        return residual + hidden
    

class AttentionBlock(nn.Module):
    def __init__(self, E, H, dropout=0.1, impl="torch", is_causal=False):
        super().__init__()
        assert E % H == 0, "head size is not multiple of embedding size"

        self.is_causal = is_causal

        self.E = E
        self.H = H
        self.dh = E//H
        self.attn_scale = 1 / (self.dh ** -0.5)

        self.preln = nn.LayerNorm(E)
        self.q_proj = LINEAR_IMPL[impl](E, E)
        self.k_proj = LINEAR_IMPL[impl](E, E)
        self.v_proj = LINEAR_IMPL[impl](E, E)
        self.o_proj = LINEAR_IMPL[impl](E, E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, L, E)
        if attn_mask is not None and self.is_causal is False:
            raise ValueError("Non-causal attention should not have an attention mask.")
        
        if self.is_causal is True and attn_mask is None:
            raise ValueError("Causal attention requires an attention mask.")
        
        B, L, E = x.shape
        residual = x
        x   = self.preln(x)
        
        q   = self.q_proj(x).view(B, L, self.H, self.dh).transpose(1, 2)                   # B, H, L, dh
        k_t = self.k_proj(x).view(B, L, self.H, self.dh).transpose(1, 2).transpose(-2, -1) # B, H, dh, L
        v   = self.v_proj(x).view(B, L, self.H, self.dh).transpose(1, 2)                   # B, H, L, dh

        score = (q @ k_t) * self.attn_scale

        if attn_mask is not None:
            # mask should be True where we want to mask out attention. e*-inf in softmax function will be zero.
            # attn_mask informs which position to fill with -inf. 
            # e*-inf in softmax function will be zero, effectively making the required positions sum to 1 after softmax.
            score = score.masked_fill(attn_mask, float('-inf')) 

        attn = F.softmax(score, dim=-1)
        prob = self.dropout(attn)

        attn = prob @ v

        # attn (B, H, L, dh) -> (B, L, H, dh) -> (B, L, E) 
        attn = attn.transpose(1, 2).contiguous().view(B, L, E)

        out = self.o_proj(attn)

        return residual + out
    
if __name__ == "__main__":
    emb_dim = 64
    expansion_dim = emb_dim*2
    num_head = 4

    tx = TransformerBlock(E=emb_dim, F=expansion_dim, H=num_head)

    print("end.")