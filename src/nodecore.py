class Node(object):
    '''Node of computational graph.

    This object represents a node of a computation graph. A node has one
    function and several arguments of that function. At the time when the node
    is generated, the function is not applied, and it is calculated as
    necessary. A node can be an argument of another node.

    Args:
        func (function): The function applied to the argument.
        args (Node, list or tuple): Arguments of the function.

    Attributes:
        data: Computation result of this Node.
    '''
    def __init__(self, func, args):
        self.func = func
        self._data = None
        _args = []
        if not isinstance(args, (tuple, list)):
            args = [args]
        for arg in args:
            if not isinstance(arg, Node):
                raise ValueError('args must be Node or list(tuple) of Nodes.')
            _args.append(arg)
        self.args = tuple(_args)

    def apply_func(self):
        expanded_args = [arg.data for arg in self.args]
        return self.func(*expanded_args)

    @property
    def data(self):
        if self._data is None:
            self._data = self.apply_func()
        return self._data


class Variable(Node):
    '''Start node of computational graph.

    Computation graph starts from this object.

    Args:
        data: value of the Variable.
    '''
    def __init__(self, data):
        self.func = None
        self.args = ()
        self._data = data

    def apply_func(self):
        raise NotImplementedError()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
