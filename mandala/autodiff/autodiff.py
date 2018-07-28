from mandala import Node
from mandala import Variable


class AutoDiff(object):
    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def __call__(self, xs):
        y = self.forward(xs)
        if isinstance(y, (tuple, list)):
            for yi in y:
                yi.creator = self
        else:
            y.creator = self
        return y


def backward(y, gy=Variable(1.)):
    xs = y.args
    gxs = y.creator.backward(xs, gy)
    for x, gx in zip(xs, gxs):
        if not isinstance(x, Node):
            return None
        if not hasattr(x, 'grad'):
            x.grad = Variable(0.)
        x.grad += gx
        if hasattr(x, 'creator'):
            backward(x, gx)


def install_node_backward():
    Node.backward = backward
