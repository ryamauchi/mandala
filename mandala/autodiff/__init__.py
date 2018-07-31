from mandala.autodiff import autodiff
from mandala.autodiff import functions


# install Node functiions.
autodiff.install_node_backward()
functions.install_node_arithmetics()
functions.install_node_get_item()
functions.install_node_reshape()
