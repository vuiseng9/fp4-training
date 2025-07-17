###
# locally implemented layers
import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, E, F, H, dropout=0.1):
        super().__init__()
        self.E = E
        self.F = F
        self.H = H

        self.attn = AttentionBlock(E, H, dropout)

        self.ffn = nn.ModuleDict({
            "preln": nn.LayerNorm(E),
            "up_proj": nn.Linear(E, F),
            "act": nn.GELU(),
            "act_dropout": nn.Dropout(dropout),
            "down_proj":nn.Linear(F, E),
            "dropout": nn.Dropout(dropout),
        })

    def forward(self, x):
        # x in (B, L, E)

        hidden = self.attn(x)
        residual = hidden

        for _, layer in self.ffn.items():
            hidden = layer(hidden)

        return residual + hidden
    

class AttentionBlock(nn.Module):
    def __init__(self, E, H, dropout=0.1):
        super().__init__()
        assert E % H == 0, "head size is not multiple of embedding size"

        self.E = E
        self.H = H
        self.dh = E//H
        self.attn_scale = 1 / (self.dh ** -0.5)

        self.preln = nn.LayerNorm(E)
        self.q_proj = nn.Linear(E, E)
        self.k_proj = nn.Linear(E, E)
        self.v_proj = nn.Linear(E, E)
        self.o_proj = nn.Linear(E, E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Image shape transformation
        # B, IC, H, W
        # B, OC(E), H, W
        B, L, E = x.shape
        residual = x
        x   = self.preln(x)
        
        q   = self.q_proj(x).reshape(B, L, self.H, self.dh).transpose(1, 2)                   # B, H, L, dh
        k_t = self.k_proj(x).reshape(B, L, self.H, self.dh).transpose(1, 2).transpose(-2, -1) # B, H, dh, L
        v   = self.v_proj(x).reshape(B, L, self.H, self.dh).transpose(1, 2)                   # B, H, L, dh

        score = (q @ k_t) * self.attn_scale
        attn = F.softmax(score, dim=-1)
        prob = self.dropout(attn)

        attn = prob @ v

        # attn (B, H, L, dh) -> (B, L, H, dh) -> (B, L, E) 
        attn = attn.transpose(1, 2).contiguous().reshape(B, L, E)

        out = self.o_proj(attn)

        return residual + out
    
if __name__ == "__main__":
    emb_dim = 64
    expansion_dim = emb_dim*2
    num_head = 4

    tx = TransformerBlock(E=emb_dim, F=expansion_dim, H=num_head)

    print("yes")