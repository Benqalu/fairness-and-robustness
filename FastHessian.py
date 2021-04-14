import torch
import numpy as np

def f(x):
	w = torch.randn(x.shape[0])
	return torch.tanh(torch.dot(w,x))


if __name__=='__main__':


	x = torch.tensor([1.,2.,3.,4.], requires_grad=True)
	print(x,f(x))
	print(torch.autograd.functional.hessian(f,x))







