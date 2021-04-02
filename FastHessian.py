import torch
import numpy as np


def jacobian(y, x, create_graph=False):
	jac = []
	flat_y = y.reshape(-1)
	grad_y = torch.zeros_like(flat_y)
	for i in range(len(flat_y)):
		grad_y[i] = 1.0
		(grad_x,) = torch.autograd.grad(
			flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
		)
		jac.append(grad_x.reshape(x.shape))
		grad_y[i] = 0.0
	return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(fx, x):
	return jacobian(jacobian(fx, x, create_graph=True), x)

if __name__=='__main__':

	def f(x):
		ret = torch.tensor(1.)
		for i in range(0,x.shape[0]):
			ret *= x[i]
		return ret

	# def f(x):
	# 	return x * x * torch.arange(4, dtype=torch.float)   

	x = torch.tensor([[1.,2.],3.,4.], requires_grad=True)
	print(x,f(x))
	print(torch.autograd.functional.hessian(f,x))







