from mandala.nodecore import Node
from mandala.nodecore import Variable


class AutoDiff(object):
    '''Base of auto-differentiable functions.'''

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, *xs):
        return self.forward(*xs).data


def backward(y, gy=Variable(1.)):
    xs = y.args
    if not hasattr(y.func, 'backward'):
        return None
    gxs = y.func.backward(*xs, gy)
    for x, gx in zip(xs, gxs):
        x.grad = gx
        backward(x, gx)


def install_node_backward():
    Node.backward = backward
