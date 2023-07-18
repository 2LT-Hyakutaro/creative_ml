import torch

w = torch.Tensor([[3., 0]])
u = torch.Tensor([[2, 0]])
b = torch.Tensor([0])
z = torch.ones((10, 2))

aff = torch.matmul(z,w.T) + b
h_1 = 1 - torch.tanh(aff)**2
psi = h_1*w
pr = torch.matmul(psi,u.T)
abs_log_det = torch.log(abs(1 + pr))
