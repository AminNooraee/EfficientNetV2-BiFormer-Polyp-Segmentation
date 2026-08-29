"""BiFormer blocks used by the decoder.

This module contains the project-specific wrapper around Bi-Level Routing
Attention. The core attention operator is imported from ``bra_legacy.py``.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

from .bra_legacy import BiLevelRoutingAttention


class RoutingTransformerBlock(nn.Module):
    """BiFormer-style residual block operating on NCHW feature maps."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.2,
        num_heads: int = 8,
        n_win: int = 2,
        qk_dim: int | None = None,
        kv_per_win: int = 4,
        kv_downsample_mode: str = "ada_avgpool",
        topk: int = 4,
        mlp_ratio: float = 4.0,
        side_dwconv: int = 5,
        before_attn_dwconv: int = 3,
    ) -> None:
        super().__init__()
        qk_dim = qk_dim or dim

        self.pos_embed = nn.Conv2d(
            dim,
            dim,
            kernel_size=before_attn_dwconv,
            padding=before_attn_dwconv // 2,
            groups=dim,
        )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = BiLevelRoutingAttention(
            dim=dim,
            num_heads=num_heads,
            n_win=n_win,
            qk_dim=qk_dim,
            kv_per_win=kv_per_win,
            kv_downsample_ratio=4,
            kv_downsample_kernel=None,
            kv_downsample_mode=kv_downsample_mode,
            topk=topk,
            param_attention="qkvo",
            param_routing=False,
            diff_routing=False,
            soft_routing=False,
            side_dwconv=side_dwconv,
            auto_pad=False,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.permute(0, 3, 1, 2)


class BiFormerDecoderBlock(nn.Module):
    """Two BiFormer-style blocks with dropout, matching the experiments."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Dropout(dropout),
            RoutingTransformerBlock(dim=dim, drop_path=drop_path),
            nn.Dropout(dropout),
            RoutingTransformerBlock(dim=dim, drop_path=drop_path),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
