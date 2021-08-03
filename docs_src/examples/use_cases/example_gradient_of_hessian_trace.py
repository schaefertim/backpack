"""Gradient of Hessian trace.
=============================
"""
from torch import autograd, cuda, device, manual_seed, randn
from torch.nn import Linear, MSELoss, Sequential

from backpack import backpack, extend
from backpack.extensions import DiagHessian, Variance

FIRST_ORDER = True

manual_seed(0)
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")

N, D = 3, 2
x = randn(N, D).to(DEVICE)
y = randn(N, 1).to(DEVICE)
model = extend(Sequential(Linear(D, 1, bias=False))).to(DEVICE)
lossfunc = extend(MSELoss(reduction="sum"))

loss = lossfunc(model(x), y)

with backpack(Variance() if FIRST_ORDER else DiagHessian()):
    loss.backward(retain_graph=True, create_graph=True)

trace = sum((p.variance if FIRST_ORDER else p.diag_h).sum() for p in model.parameters())

print(trace)

autograd.grad(trace, model.parameters())
