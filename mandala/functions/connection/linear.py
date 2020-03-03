from mandala import TensorNode
import mandala.functions as F


def blinear(inputs):
    if len(inputs) == 3:
        x, W, b = inputs
        return x @ W.T + b
    else:
        x, W = inputs
        return x @ W.T


def linear(x, W, b=None):
    if b is None:
        inputs = [x, W]
    else:
        inputs = [x, W, b]
    return TensorNode(blinear, inputs)


def dlinear(gy, *inputs):
    if len(inputs) == 3:
        x, W, b = inputs
    else:
        (x, W), b = inputs, None

    gx = gy @ W
    gW = gy.T @ x

    if b is None:
        return gx, gW

    gb = F.sum(gy, axis=0)
    return gx, gW, gb
