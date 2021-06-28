from functools import partial
from typing import Any

import jax.numpy as jnp
from chex import Array
from flax import linen as nn
from jax import random

__all__ = ["Attention", "SpatialGatingUnit", "LayerNorm"]

EPS = 1e-3
ATTN_MASK_VALUE = -1e10

Dtype = Any

LayerNorm = partial(nn.LayerNorm)


def uniform(key, scale, shape, dtype):
    return random.uniform(key, shape, dtype) * scale


class Attention(nn.Module):

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

    dim: int
    dim_out: int
    seq_len: int
    dtype: Dtype = jnp.float32

    def setup(self):
        self.norm = LayerNorm(dtype=self.dtype)
        self.proj_out = nn.Dense(features=self.dim_out, dtype=self.dtype)

    @nn.compact
    def __call__(self, x, gate_res=None) -> Array:

        x, gate = jnp.split(x, 2, axis=-1)

        gate = self.norm(gate)

        """
        # TODO: Causal Nature of SGU
        n = self.seq_len
        init_scale = EPS / n
        weights = uniform(key = random.PRNGKey(0),
            scale = init_scale, shape = (n,n),dtype = self.dtype)
        biases = jnp.ones(shape = (n,n), dtype = self.dtype)
        mask = jnp.tril(jnp.ones((n, n)))
        weights = weights * mask

        gate = np.einsum("n d, m n ->m d", gate, weights)
        gate += biases
        """

        if gate_res is not None:
            gate += gate_res

        x = x * gate
        return self.proj_out(x)
