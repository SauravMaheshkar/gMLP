from functools import partial
from typing import Any

import jax.numpy as jnp
from chex import Array
from flax import linen as nn

__all__ = ["Attention", "SpatialGatingUnit", "LayerNorm"]

ATTN_MASK_VALUE = -1e10

Dtype = Any

LayerNorm = partial(nn.LayerNorm)


class Attention(nn.Module):
    """
    Flax Module for creating an Attention Block.

    Attributes:
        dim_out: No of output dimensions
        dim_head: No of dimensions for the head
        dtype: the dtype of the computation (default: float32)
    """

    dim_out: int
    dim_head: int
    dtype: Dtype = jnp.float32

    def setup(self):
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Dense(features=self.dim_head * 3, dtype=self.dtype)
        self.to_out = nn.Dense(features=self.dim_out, dtype=self.dtype)

    @nn.compact
    def __call__(self, x) -> Array:
        n = x.shape[0]

        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        sim = jnp.einsum("i d, j d -> i j", q, k) * self.scale

        mask = jnp.triu(jnp.ones((n, n), dtype=bool), 1)
        sim = jnp.where(mask, ATTN_MASK_VALUE, sim)

        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("i j, j d -> i d", attn, v)

        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    """
    Flax Module for creating a Spatial Gating Unit.

    Attributes:
        dim_out: No of output dimensions
        dtype: the dtype of the computation (default: float32)
    """

    dim_out: int
    dtype: Dtype = jnp.float32

    def setup(self):
        self.norm = LayerNorm(dtype=self.dtype)
        self.proj_out = nn.Dense(features=self.dim_out, dtype=self.dtype)

    @nn.compact
    def __call__(self, x, gate_res=None) -> Array:

        x, gate = jnp.split(x, 2, axis=-1)

        gate = self.norm(gate)

        # TODO: Causal Nature of SGU

        if gate_res is not None:
            gate += gate_res

        x = x * gate
        return self.proj_out(x)
