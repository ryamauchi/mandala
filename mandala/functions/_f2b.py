class _F2B:

    def __init__(self):
        self.data = {}

    def add(self, bfunc, dfunc):
        self.data[bfunc] = dfunc

    def get(self, bfunc):
        if bfunc in self.data:
            return self.data[bfunc]
        # for DEBUG.
        # msg = '{} does not have backward function.'
        # print(msg.format(bfunc.__name__))
