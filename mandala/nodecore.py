class Node(object):
    '''Node of computational graph.

    This object represents a node of a computation graph. A node has one
    function and several arguments of that function. At the time when the node
    is generated, the function is not applied, and it is calculated as
    necessary. A node can be an argument of another node.

    Args:
        func (function): The function applied to the argument.
        args (Node, list or tuple): Arguments of the function.
        retain_data (bool): If True, retain computation result.
        kargs (dict): Non-node argument of function.

    Attributes:
        data: Computation result of this Node.
    '''
    def __init__(self, func, args, kargs={}, retain_data=False):
        self.func = func
        self.retain_data = retain_data
        self._data = None
        self._reference_count = 0
        self.kargs = kargs

        _args = []
        if not isinstance(args, (tuple, list)):
            args = [args]
        for arg in args:
            if not isinstance(arg, Node):
                arg = Variable(arg)
            arg._increment_ref_count()
            _args.append(arg)
        self.args = tuple(_args)

    def apply_func(self):
        expanded_args = [arg.data for arg in self.args]
        return self.func(*expanded_args, **self.kargs)

    def _increment_ref_count(self):
        self._reference_count += 1

    def _decrement_ref_count(self):
        if self._reference_count > 0:
            self._reference_count -= 1

        if self._reference_count == 0 and not self.retain_data:
            self._data = None

    def __del__(self):
        for arg in self.args:
            arg._decrement_ref_count()

    @property
    def data(self):
        if self._data is None:
            data = self.apply_func()
            self._decrement_ref_count()
            if self._reference_count > 0 or self.retain_data:
                self._data = data
        else:
            data = self._data
        return data

    @property
    def shape(self):
        # TODO: calc shape from shape of args.
        return self.data.shape


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
        self.retain_data = True
        self._reference_count = 0

    def apply_func(self):
        raise NotImplementedError()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
