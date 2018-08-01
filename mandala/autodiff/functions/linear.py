import numpy
from mandala import cuda
from mandala import Node
from mandala import Variable
from mandala.autodiff import autodiff
from mandala.autodiff import layer
from mandala.autodiff import initializers


def linear_forward(x, W, b):
    xp = cuda.get_array_module(x)
    y = xp.dot(x, W.T)
    if b is not None:
        y += b
    return y


def linear_backward_W(x, gy):
    xp = cuda.get_array_module(x)
    gW = xp.dot(gy.T, x)
    return gW


def linear_backward_b(gy):
    xp = cuda.get_array_module(gy)
    gb = xp.sum(gy, axis=0)
    return gb


def linear_backward_x(W, gy):
    xp = cuda.get_array_module(W)
    gx = xp.dot(gy, W)
    return gx


class LinearFunction(autodiff.AutoDiff):

    def forward(self, xs):
        y = Node(linear_forward, xs)
        return y

    def backward(self, xs, gy):
        x, W, b = xs
        gW = Node(linear_backward_W, [x, gy])
        gx = Node(linear_backward_x, [W, gy])
        if b is not None:
            gb = Node(linear_backward_b, [gy])
        else:
            gb = None
        return gx, gW, gb


class Linear(layer.Layer):
    def __init__(self, in_ch, out_ch, nobias=False,
                 initializer=initializers.HeNormal):
        self.W = Variable(initializer((out_ch, in_ch)))
        if nobias:
            self.b = None
        else:
            self.b = Variable(
                numpy.zeros(out_ch, dtype=numpy.float32))

    def __call__(self, x):
        return LinearFunction()([x, self.W, self.b])
