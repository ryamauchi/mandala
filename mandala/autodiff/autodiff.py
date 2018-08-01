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
    xs = y.input_nodes
    gxs = y.creator.backward(xs, gy)
    for x, gx in zip(xs, gxs):
        if not isinstance(x, Node):
            return None
        if not hasattr(x, 'grad'):
            x.grad = None
        if x.grad is None:
            x.grad = gx
        else:
            x.grad += gx
        if hasattr(x, 'creator'):
            backward(x, gx)


def cleargrads(x):
    x.grad = None


def install_node_backward():
    Node.backward = backward


def install_node_cleargrads():
    Node.cleargrads = cleargrads
