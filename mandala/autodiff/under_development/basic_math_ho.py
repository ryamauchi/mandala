from mandala import cuda
from mandala.nodecore import Node
from mandala import basic_math
from mandala.autodiff import autodiff


class Add(autodiff.AutoDiff):
    def forward(self, xs):
        y = basic_math.add(*xs)
        return y

    def backward(self, xs, gy):
        return gy, gy


def add(lhs, rhs):
    return Add()([lhs, rhs])


class Mul(autodiff.AutoDiff):
    def forward(self, xs):
        y = basic_math.mul(*xs)
        return y

    def backward(self, xs, gy):
        gx0 = mul(xs[1], gy)
        gx1 = mul(xs[0], gy)
        return gx0, gx1


def mul(lhs, rhs):
    return Mul()([lhs, rhs])


class Sub(autodiff.AutoDiff):
    def forward(self, xs):
        y = basic_math.sub(*xs)
        return y

    def backward(self, xs, gy):
        return gy, neg(gy)


def sub(lhs, rhs):
    return Sub()([lhs, rhs])


def rsub(rhs, lhs):
    return Sub()([lhs, rhs])


class Div(autodiff.AutoDiff):
    def forward(self, xs):
        y = basic_math.div(*xs)
        return y

    def backward(self, xs, gy):
        x0, x1 = xs
        gx0 = mul(div(1, x1), gy)
        gx1 = neg(mul(div(x0, (pow(x1, 2))), gy))
        return gx0, gx1


def div(lhs, rhs):
    return Div()([lhs, rhs])


def rdiv(rhs, lhs):
    return Div()([lhs, rhs])


def floordiv(lhs, rhs):
    raise NotImplementedError


def rfloordiv(rhs, lhs):
    raise NotImplementedError


class Pow(autodiff.AutoDiff):
    def forward(self, xs):
        y = basic_math.pow(*xs)
        return y

    def backward(self, xs, gy):
        x0, x1 = xs
        gx0 = mul(mul(x1, pow(x0, sub(x1, 1))), gy)
        gx1 = mul(pow(x0, x1), log(x0))
        return gx0, gx1


def pow(lhs, rhs):
    return Pow()([lhs, rhs])


def rpow(rhs, lhs):
    return Pow()([lhs, rhs])


class Log(autodiff.AutoDiff):
    def forward(self, xs):
        def _log(x):
            xp = cuda.get_array_module(x)
            return xp.log(x)
        x, = xs
        y = Node(_log, [x])
        return y

    def backward(self, xs, gy):
        x, = xs
        gx = div(1, gy)
        return gx,


def log(x):
    return Log()([x])


class Neg(autodiff.AutoDiff):

    def forward(self, xs):
        y = basic_math.neg(*xs)
        return y

    def backward(self, xs, gy):
        return neg(gy),


def neg(x):
    return Neg()([x])


def absolute(a):
    raise NotImplementedError


def matmul(lhs, rhs):
    raise NotImplementedError


def rmatmul(lhs, rhs):
    raise NotImplementedError


def install_node_arithmetics():
    Node.__neg__ = neg
    Node.__abs__ = absolute
    Node.__add__ = add
    Node.__radd__ = add
    Node.__sub__ = sub
    Node.__rsub__ = rsub
    Node.__mul__ = mul
    Node.__rmul__ = mul
    Node.__div__ = div
    Node.__truediv__ = div
    Node.__rdiv__ = rdiv
    Node.__rtruediv__ = rdiv
    Node.__floordiv__ = floordiv
    Node.__rfloordiv__ = rfloordiv
    Node.__pow__ = pow
    Node.__rpow__ = rpow
    Node.__matmul__ = matmul
    Node.__rmatmul__ = rmatmul
