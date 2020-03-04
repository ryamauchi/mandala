class _RefCount:

    def __init__(self, node):
        self.count = 0
        self.node = node

    def _check_count(self):
        if self.count == 0:
            self.node.clear_output()

    def __iadd__(self, x):
        self.count += x
        self._check_count()
        return self

    def __isub__(self, x):
        self.count -= x
        self._check_count()
        return self

    def __repr__(self):
        return '<_RefCount: count={}>'.format(self.count)


class Node:

    def __init__(self, func, inputs, *args, **kwargs):
        self.func = func
        self.inputs = inputs
        self.args = args
        self.kwargs = kwargs
        self._output = None
        self._ref_count = _RefCount(self)
        self._ref_to_inputs(1)

    def _ref_to_inputs(self, addend):
        for input_ in self.inputs:
            if not hasattr(input_, '_ref_count'):
                continue
            input_._ref_count += addend

    def _apply(self):
        _inputs = [input_.output for input_ in self.inputs]
        self._output = self.func(_inputs, *self.args, **self.kwargs)
        self._ref_to_inputs(-1)

    def clear_output(self):
        if self._output is not None:
            self._output = None
            self._ref_to_inputs(1)

    @property
    def output(self):
        if self._output is None:
            self._apply()
        return self._output

    def __eq__(self, node):
        if not isinstance(node, Node):
            return False
        if self.func != node.func:
            return False
        if self.inputs != node.inputs:
            return False
        if self.args != node.args:
            return False
        if self.kwargs != node.kwargs:
            return False
        return True

    def __del__(self):
        if self._output is None:
            self._ref_to_inputs(-1)

    def __repr__(self):
        _repr = '<{}: output={}>'.format(
            self.__class__.__name__, repr(self.output))
        return _repr


def _make_const_node(value):
    return Node(lambda inputs: value, [])


def _force_node(x):
    if isinstance(x, Node):
        return x
    return _make_const_node(x)


class TensorNode(Node):

    def __init__(self, func, inputs, *args, **kwargs):
        inputs = [_force_node(input_) for input_ in inputs]
        super(TensorNode, self).__init__(func, inputs, *args, **kwargs)


class ScalarNode(Node):

    def __init__(self, func, inputs, *args, **kwargs):
        inputs = [_force_node(input_) for input_ in inputs]
        super(ScalarNode, self).__init__(func, inputs, *args, **kwargs)
