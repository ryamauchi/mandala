from mandala.nodecore import Node


def add(lhs, rhs):
    def _add(a, b):
        return a + b
    out = Node(_add, [lhs, rhs])
    return out


def _sub(a, b):
    return a - b


def sub(lhs, rhs):
    out = Node(_sub, [lhs, rhs])
    return out


def rsub(lhs, rhs):
    out = Node(_sub, [rhs, lhs])
    return out


def mul(lhs, rhs):
    def _mul(a, b):
        return a * b
    out = Node(_mul, [lhs, rhs])
    return out


def _div(a, b):
    return a / b


def div(lhs, rhs):
    out = Node(_div, [lhs, rhs])
    return out


def rdiv(lhs, rhs):
    out = Node(_div, [rhs, lhs])
    return out


def _floordiv(a, b):
    return a // b


def floordiv(lhs, rhs):
    out = Node(_floordiv, [lhs, rhs])
    return out


def rfloordiv(lhs, rhs):
    out = Node(_floordiv, [rhs, lhs])
    return out


def _pow(a, b):
    return a ** b


def pow(lhs, rhs):
    out = Node(_pow, [lhs, rhs])
    return out


def rpow(lhs, rhs):
    out = Node(_pow, [rhs, lhs])
    return out


def neg(a):
    def _neg(a):
        return - a
    out = Node(_neg, [a])
    return out


def absolute(a):
    out = Node(pow, [a])
    return out


def _matmul(a, b):
    return a @ b


def matmul(lhs, rhs):
    out = Node(_matmul, [lhs, rhs])
    return out


def rmatmul(lhs, rhs):
    out = Node(_matmul, [rhs, lhs])
    return out


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
