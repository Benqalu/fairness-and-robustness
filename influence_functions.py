import torch
import numpy as np
from utils import get_data
from metric import Metric
from scipy.stats import pearsonr

np.set_printoptions(suppress=True)

class TorchLogistic(torch.nn.Module):
	def __init__(self, inps, bias=True, seed=None):
		super(TorchLogistic, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(inps, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()
	def forward(self, x):
		return self.activation(self.linear_model(x))

def BCELoss(y_pred,y):
	return torch.mean(y*torch.log(y_pred)+(1.0-y)*torch.log(1-y_pred))

def test(X,y):
	X=torch.tensor(X,requires_grad=False,dtype=torch.float)
	y=torch.tensor(y,requires_grad=False,dtype=torch.float)

	loss_func=torch.nn.BCELoss()
	model=TorchLogistic(inps=X.shape[1])
	optim=torch.optim.Adam(model.parameters(),lr=0.01)
	for epoch in range(0,1000):
		optim.zero_grad()
		y_pred=model(X)
		loss=loss_func(y_pred,y)
		loss.backward()
		optim.step()

	yg=y.clone().detach().requires_grad_(True)
	y_pred=model(X)
	loss=BCELoss(y_pred,yg)
	loss.backward()

	metric=Metric(true=y.reshape(-1).tolist(),pred=y_pred.reshape(-1).tolist())
	print(metric.accuracy())
	
	grad=np.array(yg.grad.tolist()).reshape(-1)
	dist=np.array((y_pred-0.5).tolist()).reshape(-1)

	maxv=0
	infindex=[]
	for i in range(0,len(grad)):
		item=grad[i]
		if item==np.inf:
			infindex.append(i)
		else:
			maxv=item if item > maxv else maxv
	maxv*=1.1
	grad[infindex]=maxv

	grad/=np.max(np.abs(grad))
	dist/=np.max(np.abs(dist))

	print('Pearson:', pearsonr(np.abs(grad),np.abs(dist))[0])

if __name__=='__main__':
	X,y=get_data('adult','race')
	X=X[:,1:]
	test(X,y)

	X,y=get_data('compas','race')
	X=X[:,1:]
	test(X,y)
