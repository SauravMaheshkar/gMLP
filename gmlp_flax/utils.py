from typing import Sequence, Type

from flax import linen as nn
from flax.linen.module import Module

__all__ = ["Sequential", "Residual", "PreNorm", "Identity"]


class Sequential(Module):
    """
    Flax Module to act as a wrapper for creating Sequential Modules
    Attributes:
        layers: A Sequence of Flax Modules
    """

    layers: Sequence[Type[Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(Module):
    """
    Flax Module to act as a wrapper for creating Residual Modules
    Attributes:
        layers: A Sequence of Flax Modules
    """

    layers: Sequence[Type[Module]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class PreNorm(Module):
    """
    Flax Module to act as a wrapper for creating Pre Normalization Modules.
    Applies Layer Normalization before each layer call.
    Attributes:
        layers: A Sequence of Flax Modules
    """

    layers: Sequence[Type[Module]]

    def setup(self):
        self.norm = nn.LayerNorm()

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = self.norm(x)
            x = layer(x)
        return x


class Identity(Module):
    """
    Flax Module to act as Identity Operation.
    """

    @nn.compact
    def __call__(self, x):
        return x
