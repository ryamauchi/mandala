from mandala import basic_math
from mandala import get_item
from mandala import cuda
from mandala.nodecore import Node
from mandala.nodecore import Variable


basic_math.install_node_arithmetics()
get_item.install_node_get_item()
cuda.install_variable_cuda()