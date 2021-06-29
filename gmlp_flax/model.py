from typing import Any

import jax.numpy as jnp
from chex import Array
from flax import linen as nn

from .layers import Attention, SpatialGatingUnit
from .utils import Identity, PreNorm, Residual, Sequential

Dtype = Any


class gMLPBlock(nn.Module):
    """
    Flax Module to create a gMLP Block with optional Attention Module
    Attributes:
        dim: No of output dimensions for the block
        dim_ff: No of output dimensions for the input projection
        attn_dim: No of dimensions for the Attention head (default: None)
        dtype: the dtype of the computation (default: float32)
    """

    dim: int
    dim_ff: int
    attn_dim: Any = None
    dtype: Dtype = jnp.float32

    def setup(self):
        self.proj_in = nn.Dense(features=self.dim_ff, dtype=self.dtype)
        self.attn = (
            Attention(
                dim_head=self.attn_dim, dim_out=self.dim_ff // 2, dtype=self.dtype
            )
            if self.attn_dim is not None
            else None
        )
        self.sgu = SpatialGatingUnit(dim_out=self.dim_ff // 2, dtype=self.dtype)
        self.proj_out = nn.Dense(features=self.dim, dtype=self.dtype)

    @nn.compact
    def __call__(self, x) -> Array:
        gate_res = self.attn(x) if self.attn is not None else None

        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.sgu(x, gate_res=gate_res)
        x = self.proj_out(x)
        return x


class gMLP(nn.Module):
    """
    Flax Module to create a gMLP Block with optional Attention Module
    Attributes:
        dim: No of output dimensions for each gMLP block
        depth: No of layers of gMLP Blocks
        num_tokens: Number of embeddings for Embedding layer (default: None)
        ff_mult: Multiplication factor for input projection of a gMLP block (default: 4)
        attn_dim: No of dimensions for the Attention head (default: None)
        dtype: the dtype of the computation (default: float32)
    """

    dim: int
    depth: int
    num_tokens: Any = None
    ff_mult: int = 4
    attn_dim: Any = None
    dtype: Dtype = jnp.float32

    def setup(self):
        dim_ff = self.dim * self.ff_mult
        self.to_embed = (
            nn.Embed(
                num_embeddings=self.num_tokens, features=self.dim, dtype=self.dtype
            )
            if self.num_tokens is not None
            else Identity()
        )

        self.layers = [
            Residual(
                [
                    PreNorm(
                        [
                            gMLPBlock(
                                dim=self.dim,
                                dim_ff=dim_ff,
                                attn_dim=self.attn_dim,
                                dtype=self.dtype,
                            )
                        ]
                    )
                ]
            )
            for i in range(self.depth)
        ]

        self.to_logits = (
            Sequential(
                [nn.LayerNorm(), nn.Dense(features=self.num_tokens, dtype=self.dtype)]
            )
            if self.num_tokens is not None
            else Identity()
        )

    @nn.compact
    def __call__(self, x) -> Array:
        x = self.to_embed(x)
        out = Sequential(self.layers)(x)
        return self.to_logits(out)
