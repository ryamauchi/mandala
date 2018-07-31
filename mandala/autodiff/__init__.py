from mandala.autodiff import autodiff
from mandala.autodiff import functions


functions.basic_math.install_node_arithmetics()
functions.get_item.install_node_get_item()
functions.reshape.install_node_reshape()

autodiff.install_node_backward()
