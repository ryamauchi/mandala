import numpy
try:
    import cupy    
    available = True
except:
    available = False


def get_array_module(*args):
    if available:
        for arg in args:
            if isinstance(arg, cupy.ndarray):
                return cupy
        return numpy
    else:
        return numpy
