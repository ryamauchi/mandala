from mandala.autodiff import autodiff
from mandala.autodiff import functions

from mandala.autodiff.functions import basic_math 
from mandala.autodiff.functions import get_item
from mandala.autodiff.functions import reshape


# install Node functiions.
autodiff.install_node_backward()
basic_math.install_node_arithmetics()
get_item.install_node_get_item()
reshape.install_node_reshape()
