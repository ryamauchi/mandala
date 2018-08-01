from mandala import cuda
from mandala import Node
from mandala.autodiff import autodiff


def sum_forward(x):
    xp = cuda.get_array_module(x)
    return xp.sum(x)


def sum_backward(x, gy):
    xp = cuda.get_array_module(x)
    return xp.ones_like(x) * gy


class SumFunction(autodiff.AutoDiff):
    def forward(self, xs):
        x = xs[0]
        y = Node(sum_forward, [x])
        return y

    def backward(self, xs, gy):
        x = xs[0]
        gx = Node(sum_backward, [x, gy])
        return gx,


def sum(x):
    return SumFunction()([x])