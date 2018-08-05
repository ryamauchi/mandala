class Node(object):
    '''Node of computational graph.

    This object represents a node of a computation graph. A node has one
    function and several arguments of that function. At the time when the node
    is generated, the function is not applied, and it is calculated as
    necessary. A node can be an argument of another node.

    Args:
        func (function): The function applied to the argument.
        input_nodes (Node, list or tuple): Argument nodes of the function.
        # retain_data (bool): If True, retain computation result.

    Attributes:
        data: Computation result of this Node.
    '''
    def __init__(self, func, input_nodes, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self._data = None
        self._reference_count = 0

        _input_nodes = []
        if not isinstance(input_nodes, (tuple, list)):
            input_nodes = [input_nodes]
        for node in input_nodes:
            if not isinstance(node, Node):
                node = Variable(node)
            node._increment_ref_count()
            _input_nodes.append(node)
        self.input_nodes = tuple(_input_nodes)

    def apply_func(self):
        expanded_nodes = [
            node.data for node in self.input_nodes]
        return self.func(*expanded_nodes, **self.kwargs)

    def _increment_ref_count(self, count=1):
        self._reference_count += count

    def _decrement_ref_count(self, count=1):
        self._reference_count = max(
            0, self._reference_count - count)

    @property
    def data(self):
        self._decrement_ref_count()
        if self._data is not None:
            data = self._data
        else:
            data = self.apply_func()

        if self._reference_count > 0:
            self._data = data
        else:
            self._data = None
        return data

    def __del__(self):
        for node in self.input_nodes:
            node._decrement_ref_count()

    def __len__(self):
        return len(self.data)

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
        self._reference_count = 0

    def apply_func(self):
        raise NotImplementedError()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
