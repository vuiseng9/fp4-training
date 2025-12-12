import os
import torch
from torch import nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock
import warnings

# just so it appears in model repr
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, ctx_size, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, ctx_size, embed_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def extra_repr(self):
        return f"L={self.weight.size(1)}, E={self.weight.size(2)}"
    
    def forward(self, x):
        # x: (B, L, E) or (B, L)
        L = x.size(1)
        return self.weight[:, :L, :]


class TinyGPT(nn.Module):
    """
    GPT with only a single transformer block
    """
    def __init__(
        self,
        vocab_size=256,
        ctx_size=128,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=2.0,
        linear_impl="torch"
    ):
        super().__init__()

        self.linear_impl = linear_impl

        self.vocab_size = vocab_size
        self.ctx_size = ctx_size

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        self.pos_embed = LearnedPositionalEmbedding(ctx_size, embed_dim)
        
        self.decoder = TransformerBlock(
            E=embed_dim,
            F=int(embed_dim * mlp_ratio),
            H=num_heads,
            impl=linear_impl,
            is_causal=True
        )

        # LM head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # tie weights
        self.lm_head.weight = self.token_embed.weight

    def _make_causal_mask(self, B, L, device):
        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device),
            diagonal=1
        )
        # (L, L) -> (1, 1, L, L); this will be broadcasted (B, H, L, L) in masked_filled
        return mask.unsqueeze(0).unsqueeze(0) # unsqueeze is almost free, just a logical view, using it for clarity

    def forward(self, x):                       
        # idx: (B, T)
        B, L = x.shape
        assert L <= self.ctx_size

        x = self.token_embed(x) + self.pos_embed(x)           # (B, L, E)

        attn_mask = self._make_causal_mask(B, L, x.device)    # (1, 1, L, L)
        x = self.decoder(x, attn_mask=attn_mask)              # (B, L, E)

        logits = self.lm_head(x)                              # (B, L, V)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, do_sample=False):
        # only support one sequence generation for simplicity
        # input_ids must be a torch tensor of shape (L,) or (1, L)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (1, L)

        # for simplicity, we also don't implement kv caching
        for _ in range(max_new_tokens):
            # crop idx to last ctx_size tokens
            ids = input_ids[:, -self.ctx_size:]         # (B, L') where L' <= ctx_size

            logits = self.forward(ids)                  # (B, L', V)
            logits = logits[:, -1, :]                   # (B, V) -- take the last token's logits

            probs = F.softmax(logits, dim=-1)                # (B, V)

            if do_sample is True:
                next_token = torch.multinomial(probs, num_samples=1)    # (B, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)

            input_ids = torch.cat((input_ids, next_token), dim=1)       # (B, L'+1)
            
        return input_ids