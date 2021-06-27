from typing import Sequence

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
