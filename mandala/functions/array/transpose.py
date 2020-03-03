from mandala import TensorNode


def btranspose(inputs, *args):
    x, = inputs
    return x.transpose(*args)


def transpose(x, *args):
    inputs = [x]
    return TensorNode(btranspose, inputs, *args)


def dtranspose(gy, x, *args):
    inv_args = []
    if args:
        inv_args = list(map(lambda x: args.index(x), range(len(args))))
    return transpose(gy, *inv_args),
