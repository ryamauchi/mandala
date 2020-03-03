import mandala

# base func
from .array.broadcast_to import bbroadcast_to  # NOQA
from .array.getitem import bgetitem  # NOQA
from .array.matmul import bmatmul # NOQA
from .array.matmul import btranspose_matmul # NOQA
from .array.numeric import bzeros_like  # NOQA
from .array.reshape import breshape  # NOQA
from .array.setitem import bsetitem  # NOQA
from .array.shape import bshape  # NOQA
from .array.sum_to import bsum_to  # NOQA
from .array.sum import bsum # NOQA
from .array.transpose import btranspose # NOQA
from .math.basic_math import bneg  # NOQA
from .math.basic_math import badd  # NOQA
from .math.basic_math import bsub  # NOQA
from .math.basic_math import bmul  # NOQA
from .connection.linear import blinear  # NOQA


# node func
from .array.broadcast_to import broadcast_to  # NOQA
from .array.getitem import getitem  # NOQA
from .array.matmul import matmul # NOQA
from .array.numeric import zeros_like  # NOQA
from .array.reshape import reshape  # NOQA
from .array.reshape import reshape_  # NOQA
from .array.setitem import setitem  # NOQA
from .array.shape import shape  # NOQA
from .array.sum_to import sum_to  # NOQA
from .array.sum import sum_ as sum # NOQA
from .array.transpose import transpose # NOQA
from .math.basic_math import neg  # NOQA
from .math.basic_math import add  # NOQA
from .math.basic_math import sub  # NOQA
from .math.basic_math import mul  # NOQA
from .math.basic_math import radd  # NOQA
from .math.basic_math import rsub  # NOQA
from .math.basic_math import rmul  # NOQA
from .connection.linear import linear  # NOQA

# multivalued node func
from .array.broadcast_to import dbroadcast_to  # NOQA
from .array.getitem import dgetitem  # NOQA
from .array.matmul import dmatmul # NOQA
from .array.matmul import dtranspose_matmul # NOQA
from .array.reshape import dreshape  # NOQA
from .array.setitem import dsetitem  # NOQA
from .array.sum_to import dsum_to  # NOQA
from .array.sum import dsum # NOQA
from .array.transpose import dtranspose # NOQA
from .math.basic_math import dneg  # NOQA
from .math.basic_math import dadd  # NOQA
from .math.basic_math import dsub  # NOQA
from .math.basic_math import dmul  # NOQA
from .connection.linear import dlinear  # NOQA

from ._f2b import _F2B

F2B = _F2B()
F2B.add(bbroadcast_to, dbroadcast_to)
F2B.add(bgetitem, dgetitem)
F2B.add(bmatmul, dmatmul)
F2B.add(btranspose_matmul, dtranspose_matmul)
F2B.add(breshape, dreshape)
F2B.add(bsetitem, dsetitem)
F2B.add(bsum_to, dsum_to)
F2B.add(bsum, dsum)
F2B.add(badd, dadd)
F2B.add(bneg, dneg)
F2B.add(bmul, dmul)
F2B.add(bsub, dsub)
F2B.add(blinear, dlinear)

del bbroadcast_to, dbroadcast_to
del bgetitem, dgetitem
del bmatmul, dmatmul
del btranspose_matmul, dtranspose_matmul
del breshape, dreshape
del bsetitem, dsetitem
del bsum_to, dsum_to
del bsum, dsum
del badd, dadd
del bneg, dneg
del bmul, dmul
del bsub, dsub
del blinear, dlinear

mandala.Node.__add__ = add
mandala.Node.__mul__ = mul
mandala.Node.__neg__ = neg
mandala.Node.__sub__ = sub
mandala.Node.__radd__ = radd
mandala.Node.__rsub__ = rsub
mandala.Node.__rmul__ = rmul
mandala.TensorNode.__getitem__ = getitem
mandala.TensorNode.__matmul__ = matmul
mandala.TensorNode.__setitem__ = setitem
mandala.TensorNode.transpose = transpose
mandala.TensorNode.T = property(transpose)
mandala.TensorNode.shape = property(shape)
mandala.TensorNode.reshape = reshape_
