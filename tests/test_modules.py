import flax.linen as nn

from gmlp_flax import gMLP
from gmlp_flax.utils import Identity, PreNorm, Residual, Sequential


def test_utils():

    seq = Sequential([nn.Dense(features=10), nn.LayerNorm()])
    res = Residual([nn.Dense(features=10), nn.LayerNorm()])
    pnorm = PreNorm([nn.Dense(features=10), nn.LayerNorm()])
    iden = Identity()

    assert isinstance(seq, nn.Module)
    assert isinstance(res, nn.Module)
    assert isinstance(pnorm, nn.Module)
    assert isinstance(iden, nn.Module)


def test_model():
    model = gMLP(num_tokens=20000, dim=512, depth=4, seq_len=1024)
    assert isinstance(model, nn.Module)
