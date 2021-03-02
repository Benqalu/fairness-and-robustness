import sys
import torch
import numpy as np
from utils import get_data
from metric import Metric

class TorchLogisticCore(torch.nn.Module):
	def __init__(self, inps, bias=True, seed=None):
		super(TorchLogisticCore, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(inps, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()
	def forward(self, x):
		return self.activation(self.linear_model(x))

class TorchLogistic(object):
	def __init__(self, lr=0.01):
		self.lr=lr
		self._n_epoch=3000

	def fit(self,X,y):
		y = torch.tensor(y, dtype=torch.float, requires_grad=False)
		X = torch.tensor(X, dtype=torch.float, requires_grad=False)
		self._utility_model=TorchLogisticCore(inps=X.shape[1])
		optim = torch.optim.Adam(
			self._utility_model.parameters(), lr=self.lr
		)
		loss_func=torch.nn.BCELoss()
		for epoch in range(1, self._n_epoch):
			optim.zero_grad()
			outs=self._utility_model(X)
			loss=loss_func(outs,y)
			loss.backward()
			optim.step()
		metric=Metric(true=y.numpy(), pred=outs.detach().numpy())
		print('Training accuracy: %.4f'%metric.accuracy())

	def predict(self,X,y=None,s=None):
		X = torch.tensor(X, dtype=torch.float, requires_grad=False)
		if y is None:
			return self._utility_model(X).reshape(-1).tolist()
		else:
			metric=Metric(pred=self._utility_model(X).reshape(-1).tolist(), true=y.reshape(-1))
			print(metric.accuracy())
			if s is not None:
				print(metric.positive_disparity(s=s.reshape(-1)))
		

X,y=get_data('adult','sex')
s=X[:,0]
X=X[:,1:]
f=open('transformed_X.txt')
X_=np.array(eval(f.readline()))
f.close()

model=TorchLogistic()
model.fit(X_,y)
model.predict(X,y=y,s=s)


