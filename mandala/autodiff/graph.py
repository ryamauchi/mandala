from mandala.autodiff.layer import Layer


class Graph(object):
    def __init__(self):
        self.params = {}
        self.subgraphs = {}

    def __setattr__(self, key, value):
        if isinstance(value, (Layer, Graph)):
            for _key, _value in value.params.items():
                self.params[key + '/' + _key] = _value

            self.subgraphs[key] = value

        object.__setattr__(self, key, value)

    def to_gpu(self):
        for param in self.params.values():
            param.to_gpu()

    def to_cpu(self):
        for param in self.params.values():
            param.to_cpu()

    def cleargrads(self):
        for subgraph in self.subgraphs.values():
            subgraph.cleargrads()
