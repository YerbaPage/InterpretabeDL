import torch
from hessian import hessian

x = torch.tensor([1.5, 2.5], requires_grad=True)
aaa = sum(x.pow(2))
h = hessian(aaa, x, create_graph=True)

print(h)
# tensor([[12.5, 15],
#         [15,  4.5]], grad_fn=<CopySlices>)

# h2 = hessian(h.sum(), x)
# print(h2)