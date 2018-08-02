class Node(object):
    '''Node of computational graph.

    This object represents a node of a computation graph. A node has one
    function and several arguments of that function. At the time when the node
    is generated, the function is not applied, and it is calculated as
    necessary. A node can be an argument of another node.

    Args:
        func (function): The function applied to the argument.
        input_nodes (Node, list or tuple): Argument nodes of the function.
        retain_data (bool): If True, retain computation result.

    Attributes:
        data: Computation result of this Node.
    '''
    def __init__(self, func, input_nodes, retain_data=False, **kwargs):
        self.func = func
        self.retain_data = retain_data
        self._data = None
        self._reference_count = 0
        self.kwargs = kwargs
        self.reserved = False

        _input_nodes = []
        if not isinstance(input_nodes, (tuple, list)):
            input_nodes = [input_nodes]
        for node in input_nodes:
            if not isinstance(node, Node):
                node = Variable(node)
            _input_nodes.append(node)
        self.input_nodes = tuple(_input_nodes)

    def apply_func(self):
        expanded_nodes = [node.data for node in self.input_nodes]
        return self.func(*expanded_nodes, **self.kwargs)

    def _increment_ref_count(self):
        if self._reference_count == 0:
            for node in self.input_nodes:
                node._increment_ref_count()
        self._reference_count += 1

    def _decrement_ref_count(self):
        if self._reference_count == 1:
            for node in self.input_nodes:
                node._decrement_ref_count()
            if not self.retain_data:
                self._data = None
        if self._reference_count > 0:
            self._reference_count -= 1

    def __del__(self):
        if self.reserved:
            for node in self.input_nodes:
                node._decrement_ref_count()

    def __len__(self):
        return len(self.data)

    def reserve(self):
        self.reserved = True
        self._increment_ref_count()

    def unreserve(self):
        self.reserved = False
        self._decrement_ref_count()

    @property
    def data(self):
        if self._data is None:
            data = self.apply_func()
            if self.reserved:
                self._decrement_ref_count()
            if (self._reference_count > 0
                    or self.retain_data):
                self._data = data
        else:
            data = self._data
        return data

    @property
    def shape(self):
        # TODO: calc shape from shape of input_nodes.
        return self.data.shape


class Variable(Node):
    '''Start node of computational graph.

    Computation graph starts from this object.

    Args:
        data: value of the Variable.
    '''
    def __init__(self, data):
        self.func = None
        self.input_nodes = ()
        self._data = data
        self.retain_data = True
        self._reference_count = 0
        self.reserved = False

    def apply_func(self):
        raise NotImplementedError()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
