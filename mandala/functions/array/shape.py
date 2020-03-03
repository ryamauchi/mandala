from mandala import TensorNode


def bshape(inputs):
    x, = inputs
    return x.shape


def shape(x):
    return TensorNode(bshape, [x])
