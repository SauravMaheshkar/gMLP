from typing import Any

from chex import Array
from flax import linen as nn

from .layers import Attention, SpatialGatingUnit
from .utils import Identity, PreNorm, Residual, Sequential, dropout_layers


class gMLPBlock(nn.Module):

    dim: int
    dim_ff: int
    seq_len: int
    attn_dim: Any = None

    def setup(self):
        self.proj_in = Sequential(nn.Dense(features=self.dim_ff), nn.gelu())
        self.attn = (
            Attention(dim_head=self.attn_dim, dim_out=self.dim_ff // 2)
            if self.attn_dim is not None
            else None
        )
        self.sgu = SpatialGatingUnit(
            dim=self.dim_ff, dim_out=self.dim_ff // 2, seq_len=self.seq_len
        )
        self.proj_out = nn.Dense(features=self.dim)

    @nn.compact
    def __call__(self, x) -> Array:
        gate_res = self.attn(x) if self.attn is not None else None

        x = self.proj_in(x)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


class gMLP(nn.Module):

    dim: int
    depth: int
    seq_len: int
    num_tokens: Any = None
    ff_mult: int = 4
    attn_dim: Any = None
    prob_survival: float = 1.0

    def setup(self):
        dim_ff = self.dim * self.ff_mult
        self.to_embed = (
            nn.Embed(num_embeddings=self.num_tokens, features=self.dim)
            if self.num_tokens is not None
            else Identity()
        )

        self.layers = [
            Residual(
                PreNorm(
                    gMLPBlock(
                        dim=self.dim,
                        dim_ff=dim_ff,
                        seq_len=self.seq_len,
                        attn_dim=self.attn_dim,
                    )
                )
            )
            for i in range(self.depth)
        ]

        self.to_logits = (
            Sequential(nn.LayerNorm(), nn.Dense(features=self.num_tokens))
            if self.num_tokens is not None
            else Identity()
        )

    @nn.compact
    def __call__(self, x) -> Array:
        x = self.to_embed(x)
        layers = (
            self.layers
            if not self.training
            else dropout_layers(self.layers, self.prob_survival)
        )
        out = Sequential(*layers)(x)
        return self.to_logits(out)
