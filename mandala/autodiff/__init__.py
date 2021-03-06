from mandala.autodiff import autodiff
from mandala.autodiff import functions
from mandala.autodiff.graph import Graph  # NOQA
from mandala.autodiff.autodiff import AutoDiff  # NOQA


# install Node functiions.
autodiff.install_node_backward()
autodiff.install_node_cleargrads()
functions.install_node_arithmetics()
functions.install_node_get_item()
functions.install_node_reshape()
