from mandala import Variable


class Layer(object):
    '''Base class of NN Layers.'''

    def __setattr__(self, key, value):
        if isinstance(value, Variable):
            if not hasattr(self, 'params'):
                self.params = {}
            self.params[key] = value

        object.__setattr__(self, key, value)

    def to_gpu(self):
        for param in self.params.values():
            param.to_gpu()

    def to_cpu(self):
        for param in self.params.values():
            param.to_cpu()

    def to_cpu(self):
        for param in self.params.values():
            param.to_cpu()

    def cleargrads(self):
        for param in self.params.values():
            param.cleargrads()
