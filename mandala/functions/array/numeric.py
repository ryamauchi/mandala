from mandala import TensorNode
from .utils import get_array_module


def bzeros_like(inputs, xp):
    x, = inputs
    if xp == 'auto':
        xp = get_array_module(x)
    return xp.zeros_like(x)


def zeros_like(x, xp='auto'):
    return TensorNode(bzeros_like, [x], xp)
