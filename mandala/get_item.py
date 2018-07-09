from mandala.nodecore import Node
from mandala.nodecore import Variable


# support function
def _make_node(func, args):
    _args = []
    for arg in args:
        if not isinstance(arg, Node):
            arg = Variable(arg)
        _args.append(arg)
    return Node(func, _args)


def get_item(x, slices):
    def _get_item(x, slices):
        return x[slices]
    out = _make_node(_get_item, [x, slices])
    return out


def install_node_get_item():
    Node.__getitem__ = get_item
