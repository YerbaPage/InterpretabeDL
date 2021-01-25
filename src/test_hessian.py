import torch
from hessian import *

x = torch.tensor([1.5, 2.5], requires_grad=True)
aaa = sum(x.pow(2))

g = jacobian(aaa, x)
h = hessian(aaa, x, create_graph=True)
print(g)
print(h)

grad = torch.autograd.grad(aaa, x, retain_graph=True, create_graph=True)
# hess = torch.autograd.grad(torch.autograd.grad(aaa, x, retain_graph=True, create_graph=True), x, retain_graph=True,  create_graph=True)
hess = jacobian(grad, x)
print(grad)
print(hess)

# tensor([[12.5, 15],
#         [15,  4.5]], grad_fn=<CopySlices>)

# h2 = hessian(h.sum(), x)
# print(h2)