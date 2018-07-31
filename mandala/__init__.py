from mandala.nodecore import Node   # NOQA
from mandala.nodecore import Variable   # NOQA

from mandala import functions
from mandala import cuda


functions.basic_math.install_node_arithmetics()
functions.get_item.install_node_get_item()
functions.reshape.install_node_reshape()
cuda.install_variable_cuda()
