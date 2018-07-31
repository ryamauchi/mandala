import cupy
from mandala import cuda
from mandala import Node
from mandala.autodiff import autodiff


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cupy.cuda.cudnn.CUDNN_ACTIVATION_RELU


def forward_relu(x):
    
    xp = cuda.get_array_module(x)

    if xp == cupy and cuda.cudnn_enable:
        y = cudnn.activation_forward(x, _mode)
    else:
        y = xp.maximum(x, 0)

    return y


def backward_relu(x, y, gy):
    xp = cuda.get_array_module(x)

    if xp == cupy and cuda.cudnn_enable:
        gx = cudnn.activation_backward(x, y, gy, _mode)
    else:
        gx = (x > 0) * gy

    return gx


class ReLUFunction(autodiff.AutoDiff):

    def forward(self, xs):
        x, = xs
        self.y = Node(forward_relu, [x])
        return self.y

    def backward(self, xs, gy):
        x, = xs
        gx = Node(backward_relu, [x, self.y, gy])
        return gx,


def relu(x):
    return ReLUFunction()([x])
