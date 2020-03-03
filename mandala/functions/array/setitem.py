import copy as copy_
from mandala import TensorNode
import mandala.functions as F


def bsetitem(inputs, copy):
    x, i, v = inputs
    if copy:
        x = copy_.deepcopy(x)
    x[i] = v
    return x


def setitem(x, i, v, copy=True):
    inputs = [x, i, v]
    return TensorNode(bsetitem, inputs, copy)


def dsetitem(gy, x, i, v, copy):
    gx = setitem(gy, i, 0., copy=True)
    gv = F.getitem(gy, i)
    return gx, None, gv
