from mandala import cuda
from mandala import Node
from mandala.autodiff import autodiff


def get_item_forward(x, slices):
    return x[slices]


def get_item_backward(x, slices, gy):
    xp = cuda.get_array_module(x)
    gx = xp.zeros_like(x)
    gx[slices] = gy
    return gx


class GetItem(autodiff.AutoDiff):
    def forward(self, xs):
        x, slices = xs
        y = Node(get_item_forward, [x, slices])
        return y

    def backward(self, xs, gy):
        x, slices = xs
        gx = Node(get_item_backward, [x, slices, gy])
        return gx,


def get_item(x, slices):
    return GetItem()([x, slices])


def install_node_get_item():
    Node.__getitem__ = get_item
