from mandala import Variable
from mandala import ScalarVariable
from mandala.functions import F2B


def backward(y, gy=ScalarVariable(1.)):

    if isinstance(y, Variable):
        if y.grad is None:
            y.grad = gy
        else:
            y.grad += gy

    func = y.func
    inputs = y.inputs
    args = y.args
    kwargs = y.kwargs
    gfunc = F2B.get(func)

    if gfunc is not None and len(inputs) != 0:
        ginputs = gfunc(gy, *inputs, *args, **kwargs)

        for x, gx in zip(inputs, ginputs):
            if gx is None:
                continue
            backward(x, gx)
