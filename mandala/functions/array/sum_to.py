from mandala import TensorNode
import mandala.functions as F


def bsum_to(inputs):
    x, shape = inputs
    if x.shape == shape:
        return x
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def sum_to(x, shape):
    inputs = [x, shape]
    return TensorNode(bsum_to, inputs)


def dsum_to(gy, x, shape):
    x_shape = F.shape(x)
    gx = F.broadcast_to(gy, x_shape)
    return gx, None
