from mandala import TensorNode
import mandala.functions as F


def bgetitem(inputs):
    x, i = inputs
    return x[i]


def getitem(x, i):
    inputs = [x, i]
    return TensorNode(bgetitem, inputs)


def dgetitem(gy, x, index):
    gx = F.zeros_like(x)
    gx = F.setitem(gx, index, gy, copy=False)
    return gx, None
