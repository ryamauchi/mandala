from mandala import TensorNode
import mandala.functions as F
from .utils import get_array_module


def bbroadcast_to(inputs):
    x, shape = inputs
    xp = get_array_module(x)
    y = xp.broadcast_to(x, shape)
    return y


def broadcast_to(x, shape):
    inputs = [x, shape]
    return TensorNode(bbroadcast_to, inputs)


def dbroadcast_to(gy, x, shape):
    x_shape = F.shape(x)
    gx = F.sum_to(gy, x_shape)
    return gx, None
