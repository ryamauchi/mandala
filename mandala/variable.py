from mandala import Node
from mandala import TensorNode
from mandala import ScalarNode


class Variable(Node):

    def __init__(self, value):
        self.value = value
        self.grad = None
        super(Variable, self).__init__(lambda inputs: self.value, [])


class TensorVariable(Variable, TensorNode):

    def __init__(self, value):
        super(TensorVariable, self).__init__(value)


class ScalarVariable(Variable, ScalarNode):

    def __init__(self, value):
        super(ScalarVariable, self).__init__(value)
