from mandala.nodecore import Node
from mandala.autodiff import autodiff


def reshape_forward(x, shape):
    return x.reshape(shape)


def reshape_backward(x_shape, gy):
    return gy.reshape(x_shape)


class ReshapeFunction(autodiff.AutoDiff):

    def forward(self, xs):
        x, shape = xs
        self.x_shape = x.shape
        y = Node(reshape_forward, [x, shape])
        return y

    def backward(self, xs, gy):
        gx = Node(reshape_backward, [self.x_shape, gy])
        return gx


def reshape(x, shape):
    return ReshapeFunction()([x, shape])
