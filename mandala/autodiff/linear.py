import numpy
from mandala import cuda
from mandala.nodecore import Node
from mandala.nodecore import Variable
from mandala.autodiff import autodiff
from mandala.autodiff import linear
from mandala.autodiff import initializers


def linear_forward(x, W, b):
    xp = cuda.get_array_module(x)
    y = xp.matmul(x, W.T)
    if b is not None:
        y += b
    return y


def linear_backward_W(x, gy):
    xp = cuda.get_array_module(x)
    gW = xp.matmul(gy.T, x)
    return gW


def linear_backward_b(b, gy):
    xp = cuda.get_array_module(b)
    gb = gy.sum(axis=0)
    return gb


def linear_backward_x(W, gy):
    xp = cuda.get_array_module(W)
    gx = xp.matmul(gy, W)
    return gx


class LinearFunction(autodiff.AutoDiff):

    def forward(self, xs):
        x, W, b = xs
        y = Node(linear_forward, [x, W, b])
        return y

    def backward(self, xs, gy):
        x, W, b = xs
        gW = Node(linear_backward_W, [x, gy])
        gx = Node(linear_backward_x, [W, gy])
        if b is not None:
            gb = Node(linear_backward_b, [W, gy])
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
            self.b = Variable(numpy.zeros(out_ch, dtype=np.float32))

    def __call__(self, x):
        return LinearFunction()([x, self.W, self.b])
