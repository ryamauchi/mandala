import numpy
from mandala import Variable

available = False
cudnn_enable = False

try:
    import cupy
    available = True
except Exception as e:
    _resolution_error = e

if available:
    try:
        import cupy.cudnn
        cudnn_enable = True
        cudnn = cupy.cudnn
    except Exception as e:
        _resolution_error = e


def get_array_module(*args):
    if available:
        for arg in args:
            if isinstance(arg, cupy.ndarray):
                return cupy
        return numpy
    else:
        return numpy


def to_cpu(x):
    if available and isinstance(x, cupy.ndarray):
        return x.get()
    else:
        return x


def to_gpu(x, device=None):
    if not available:
        msg = 'CUDA environment is not correctly set up'
        raise RuntimeError(msg)
    with cupy.cuda.Device(device):
        x_gpu = cupy.array(x)
    return x_gpu


def _to_cpu_variable(x):
    x.data = to_cpu(x.data)


def _to_gpu_variable(x):
    x.data = to_gpu(x.data)


def install_variable_cuda():
    Variable.to_cpu = _to_cpu_variable
    Variable.to_gpu = _to_gpu_variable
