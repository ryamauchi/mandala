import numpy
from mandala import cuda
from mandala import Node
from mandala import Variable
from mandala.autodiff import autodiff
from mandala.autodiff import layer
from mandala.autodiff import initializers
from mandala.autodiff.utils_conv import im2col
from mandala.autodiff.utils_conv import col2im


def convolution_2d_forward(x, W, b, sy, sx, ph, pw,
                           cover_all, dy, dx, cls):
    xp = cuda.get_array_module(x)
    
    _, _, kh, kw = W.shape
    col = im2col(x, kh, kw, sy, sx, ph, pw,
                 cover_all=cover_all, dy=dy, dx=dx)
    cls.col = col

    y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
    if b is not None:
        y += b
    y = xp.rollaxis(y, 3, 1)
    return y


def convolution_2d_backward_x(x, W, gy, sy, sx, ph, pw):
    xp = cuda.get_array_module(W)

    _, _, h, w = x.shape
    gcol = xp.tensordot(W, gy, (0, 1))
    gcol = xp.rollaxis(gcol, 3)
    gx = col2im(gcol, sy, sx, ph, pw, h, w)
    return gx


def convolution_2d_backward_W(col, gy):
    xp = cuda.get_array_module(col)
    gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
    return gW


def convolution_2d_backward_b(gy):
    gb = gy.sum(axis=(0, 2, 3))
    return gb


class Convolution2DFunction(autodiff.AutoDiff):

    def __init__(self, stride=1, pad=0, cover_all=False,
                 dilate=1):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.dy, self.dx = _pair(dilate)
        self.cover_all = cover_all

    def forward(self, xs):
        x, W, b = xs
        y = Node(convolution_2d_forward, xs,
                 sy=self.sy, sx=self.sx, ph=self.ph, pw=self.pw,
                 cover_all=self.cover_all, dy=self.dy, dx=self.dx,
                 cls=self)
        return y

    def backward(self, xs, gy):
        x, W, b = xs
        col = self.col

        gx = Node(convolution_2d_backward_x, [x, W, gy],
                  sy=self.sy, sx=self.sx, ph=self.ph, pw=self.pw)
        gW = Node(convolution_2d_backward_W, [col, gy])
        if b is None:
            gb = None
        else:
            gb = Node(convolution_2d_backward_b, [gy])
        return gx, gW, gb


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class Convolution2D(layer.Layer):
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=0,
                 nobias=False, cover_all=False, dilate=1):

        self.config = {
            'ksize': _pair(ksize),
            'stride': _pair(stride),
            'pad': _pair(pad),
            'dilate': _pair(dilate),
            'cover_all': cover_all
        }

        kh, kw = self.config['ksize']
        self.W = Variable(
            initializers.HeNormal((out_ch, in_ch, kh, kw)))
        if nobias:
            self.b = None
        else:
            self.b = Variable(
                numpy.zeros(out_ch, dtype=numpy.float32))

    def __call__(self, x):
        xs = [x, self.W, self.b]

        stride = self.config['stride']
        pad = self.config['pad']
        cover_all = self.config['cover_all']
        dilate = self.config['dilate']

        return Convolution2DFunction(
            stride, pad, cover_all, dilate)(xs)
