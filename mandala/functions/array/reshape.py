from mandala import TensorNode
import mandala.functions as F


def breshape(inputs):
    x, shape = inputs
    return x.reshape(shape)


def reshape(x, shape):
    inputs = [x, shape]
    return TensorNode(breshape, inputs)


def dreshape(gy, x, shape):
    x_shape = F.shape(x)
    return reshape(gy, x_shape)


def reshape_(x, *args):
    return reshape(x, args)
