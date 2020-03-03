from mandala import TensorNode
import mandala.functions as F


def bsum(inputs, axis, keepdims):
    x, = inputs
    return x.sum(axis=axis, keepdims=keepdims)


def sum_(x, axis=None, keepdims=False):
    inputs = [x]
    return TensorNode(bsum, inputs, axis, keepdims)


def dsum(gy, x, axis, keepdims):
    x_shape = F.shape(x)
    gx = F.broadcast_to(gy, x_shape)
    return gx,
