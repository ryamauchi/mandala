from mandala.nodecore import Node


def reshape(x, shape):
    def _reshape(x, slice):
        return x.reshape(shape)
    out = Node(_reshape, [x, shape])
    return out


def _reshape(x, *shape):
    return reshape(x, shape)


def install_node_reshape():
    Node.reshape = _reshape
