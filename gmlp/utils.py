from random import randrange
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

__all__ = ["Sequential", "Residual", "PreNorm", "Identity"]


class Sequential(nn.Module):
    """
    Flax Module to act as a wrapper for creating Sequential Modules
    Attributes:
        layers: A Sequence of Flax Modules
    """

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(nn.Module):
    """
    Flax Module to act as a wrapper for creating Residual Modules
    Attributes:
        layers: A Sequence of Flax Modules
    """

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class PreNorm(nn.Module):

    layers: Sequence[nn.Module]

    def setup(self):
        self.norm = nn.LayerNorm()

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = self.norm(x)
            x = layer(x)
        return x


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = jnp.zeros(num_layers).uniform_(0.0, 1.0) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers
