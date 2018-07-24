from mandala.autodiff import autodiff
from mandala.autodiff import basic_math
from mandala.autodiff import get_item


autodiff.install_node_backward()
basic_math.install_node_arithmetics()
get_item.install_node_get_item()
