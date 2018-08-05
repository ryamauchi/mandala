import sys
import numpy as np
import time

try:
    import cupy as cp
    xp = cp
except:
    xp = np

sys.path.append('../')

import mandala
from mandala import Node
from mandala import Variable

import mandala.autodiff as ad
import mandala.autodiff.functions as F


W = xp.arange(15, dtype=np.float32).reshape(3, 5)
b = xp.arange(3, dtype=np.float32)

batch_size = 128


class Model(ad.Graph):
    def __init__(self):
        super(Model, self).__init__()

        self.l0 = F.Linear(   5, 1000)
        self.l1 = F.Linear(1000, 1000)
        self.l2 = F.Linear(1000, 1000)
        self.l3 = F.Linear(1000, 1000)
        self.l4 = F.Linear(1000, 1000)
        self.l5 = F.Linear(1000, 1000)
        self.l6 = F.Linear(1000,    3)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        y = F.relu(self.l6(h))
        return y
    

model = Model()
if not xp == np:
    model.to_gpu()


s = time.time()
lr = 1e-4

for i in range(10000):
    # make batch
    x = Variable(xp.random.random((batch_size, 5)).astype(np.float32))
    t = Variable(xp.matmul(x.data, W.T) + b)
    
    # forward
    y = model(x)
    loss = (y - t) ** 2 / batch_size

    # loss
    loss = F.sum((y - t) ** 2) / batch_size
    # backward
    model.cleargrads()
    loss.backward()
    
    del loss
    del y

    for p in model.params.values():
        p.data -= lr * p.grad.data
        del p.grad

print('time:', time.time() - s)