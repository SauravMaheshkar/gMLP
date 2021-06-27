from functools import partial
from typing import Any

import jax.numpy as jnp
from flax import linen as nn

__all__ = ["Attention", "SGU"]

EPS = 1e-3
ATTN_MASK_VALUE = -1e10

Dtype = Any

LayerNorm = partial(nn.LayerNorm, create_scale=True, create_offset=True, axis=-1)


class Attention(nn.Module):

    dim_out: int
    dim_head: int
    dtype: Dtype = jnp.float32

    def setup(self):
        self.scale = self.dim_head**-0.5
        self.to_qkv = nn.Dense(features=self.dim_head * 3, dtype=self.dtype)
        self.to_out = nn.Dense(features=self.dim_out, dtype=self.dtype)

    @nn.compact
    def __call__(self, x):
        n = x.shape[0]

        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        sim = jnp.einsum("i d, j d -> i j", q, k) * self.scale

        mask = jnp.triu(jnp.ones((n, n), dtype=bool), 1)
        sim = jnp.where(mask, ATTN_MASK_VALUE, sim)

        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("i j, j d -> i d", attn, v)

        return self.to_out(out)


class SGU(nn.Module):

    dim: int
    dim_out: int
    seq_len: int
    dtype: Dtype = jnp.float32

    def setup(self):
        self.norm = LayerNorm()
        self.proj_out = nn.Dense(features=self.dim_out, dtype=self.dtype)

    @nn.compact
    def __call__(self, x, gate_res=None):
        n = self.seq_len

        x, gate = jnp.split(x, 2, axis=-1)

        gate = self.norm(gate)
        init_scale = EPS / n
        init_eps = nn.initializers.uniform(scale=init_scale, dtype=self.dtype)

        weights = self.param("spatial_weights", init_eps, (n, n))
        biases = self.param("spatial_biases", jnp.ones(), (n, 1))

        mask = jnp.tril(jnp.ones((n, n)))
        weights = weights * mask

        gate = jnp.einsum("n d, m n -> m d", gate, weights)
        gate += biases

        if gate_res is not None:
            gate += gate_res

        x = x * gate
        return self.proj_out(x)
