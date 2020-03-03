from mandala import TensorNode


def btranspose_matmul(inputs):
    x, = inputs
    ndim = len(x.shape)
    axis = list(range(ndim))
    axis[-2:] = axis[:-3:-1]
    return x.transpose(*axis)


def transpose_matmul(x):
    return TensorNode(btranspose_matmul, [x])


def dtranspose_matmul(gy, x):
    return transpose_matmul(gy),


def bmatmul(inputs):
    x0, x1 = inputs
    return x0 @ x1


def matmul(x0, x1):
    inputs = [x0, x1]
    return TensorNode(bmatmul, inputs)


def dmatmul(gy, x0, x1):
    gx0 = matmul(gy, transpose_matmul(x1))
    gx1 = matmul(transpose_matmul(x0), gy)
    return gx0, gx1
