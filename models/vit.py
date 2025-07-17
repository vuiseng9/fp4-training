import os
import torch
from torch import nn
import torch.nn.functional as F
from .impl_layer import TransformerBlock

USE_TORCHVISION_VIT = int(os.getenv('USE_TORCHVISION_VIT', 0))

class TinyViT(nn.Module):
    """
    ViT with only a single transformer block
    """
    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_chans=1,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=2.0,
    ):
        super().__init__()

        assert img_size % patch_size == 0, "patch_size must divide img_size exactly"
        self.num_patches = (img_size // patch_size) ** 2

        # (a) Patch → embedding (conv equals linear on flattened patch)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

        # (b) Class token + learnable sinusoidal-ish positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1 + self.num_patches, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # (c) Single Transformer encoder block
        if USE_TORCHVISION_VIT == 1:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    batch_first=True,
                ),
                num_layers=1
            )
        elif USE_TORCHVISION_VIT == 0:
            self.encoder = TransformerBlock(
                E=embed_dim,
                F=int(embed_dim * mlp_ratio),
                H=num_heads
            )
        else:
            raise ValueError("Invalid USE_TORCHVISION_VIT value. Must be 0 or 1.")

        # (d) Classification head
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):                       # x: (B,1,28,28)
        B = x.size(0)
        x = self.patch_embed(x)                 # (B, embed_dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)        # (B, 16, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B,17,embed_dim)
        x = x + self.pos_embed                         # add position info

        x = self.encoder(x)                            # (B,17,embed_dim)
        cls_out = x[:, 0]                              # [CLS] token
        return self.mlp_head(cls_out)