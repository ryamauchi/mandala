import numpy


def HeNormal(shape, scale=1.0):
    fan_in = numpy.prod(shape[1:])
    std = scale * numpy.sqrt(2 / fan_in)
    init_W = numpy.random.normal(0, std, shape)
    return init_W.astype(numpy.float32)
