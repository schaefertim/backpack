"""Gradient of Hessian trace.
=============================
"""
from torch import autograd, cuda, device, manual_seed, randn, zeros
from torch.nn import Linear, MSELoss, Sequential

from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagGGNExact, DiagHessian, Variance

FIRST_ORDER = True

manual_seed(0)
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")

N, D = 3, 2
x = randn(N, D, requires_grad=True).to(DEVICE)
y = randn(N, 1).to(DEVICE)
model = extend(Sequential(Linear(D, D, bias=False), Linear(D, 1, bias=False))).to(
    DEVICE
)
lossfunc = extend(MSELoss(reduction="sum"))

loss = lossfunc(model(x), y)

extensions = (BatchGrad(), Variance(), DiagHessian(), DiagGGNExact())
print("\nBackward pass.")
with backpack(*extensions):
    loss.backward(retain_graph=True, create_graph=True)

print("\nParameter analysis.")
for p in model.parameters():
    for extension in extensions:
        quantity = getattr(p, extension.savefield)
        if isinstance(quantity, list):
            quantity = quantity[0]
        print(f"{extension} grad_fn: {quantity.grad_fn}")

print("\nCalculate some derivative.")
for extension in extensions:
    print(f"for extension {extension}")
    sum_all = zeros(1, device=DEVICE)
    for p in model.parameters():
        quantity = getattr(p, extension.savefield)
        if isinstance(quantity, list):
            quantity = quantity[0]
        sum_all += quantity.sum()
    print(f"\tsum_all = {sum_all}")
    derivative = autograd.grad(sum_all, x, retain_graph=True)[0]
    print(f"\tderivative wrt x = {derivative}")
