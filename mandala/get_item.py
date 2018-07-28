from mandala import Node


def get_item(x, slices):
    def _get_item(x, slices):
        return x[slices]
    out = Node(_get_item, [x, slices])
    return out


def install_node_get_item():
    Node.__getitem__ = get_item
