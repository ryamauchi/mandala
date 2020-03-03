from mandala import TensorNode
from mandala import ScalarNode


def _get_node_class(nodes):
    for node in nodes:
        if isinstance(node, TensorNode):
            return TensorNode
    return ScalarNode


def bneg(inputs):
    x, = inputs
    return -x


def badd(inputs):
    x0, x1 = inputs
    return x0 + x1


def bmul(inputs):
    x0, x1 = inputs
    return x0 * x1


def bsub(inputs):
    x0, x1 = inputs
    return x0 - x1


def neg(x):
    inputs = [x]
    Node_ = _get_node_class(inputs)
    return Node_(bneg, inputs)


def add(x0, x1):
    inputs = [x0, x1]
    Node_ = _get_node_class(inputs)
    return Node_(badd, inputs)


def mul(x0, x1):
    inputs = [x0, x1]
    Node_ = _get_node_class(inputs)
    return Node_(bmul, inputs)


def sub(x0, x1):
    inputs = [x0, x1]
    Node_ = _get_node_class(inputs)
    return Node_(bsub, inputs)


def radd(x0, x1):
    return add(x1, x0)


def rsub(x0, x1):
    return sub(x1, x0)


def rmul(x0, x1):
    return mul(x1, x0)


def dneg(gy, x):
    gx = neg(gy)
    return gx,


def dadd(gy, x0, x1):
    gx0 = gy
    gx1 = gy
    return gx0, gx1


def dmul(gy, x0, x1):
    gx0 = mul(x1, gy)
    gx1 = mul(x0, gy)
    return gx0, gx1


def dsub(gy, x0, x1):
    gx0 = gy
    gx1 = neg(gy)
    return gx0, gx1
