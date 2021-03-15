import torch
import numpy as np
from metric import Metric

class TorchNNCore(torch.nn.Module):
	def __init__(self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU):
		super(TorchNNCore, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		struct = [inps]+hiddens+[1]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i], bias=bias))
			if i==len(struct)-1:
				layers.append(torch.nn.Sigmoid())
			else:
				layers.append(hidden_activation())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		return output

class TorchNeuralNetworks(object):
	def __init__(self, lr=0.01, n_epoch=500, hiddens=[], seed=None):
		self._lr=lr
		self._n_epoch=n_epoch
		self._model=None
		self._loss_func=torch.nn.BCELoss(reduction='none')
		self._hiddens=hiddens
		self._seed=seed
		if self._seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
	def fit(self,X,s,y,weight=None):
		X=torch.tensor(X, dtype=torch.float)
		y=torch.tensor(y.reshape(-1,1), dtype=torch.float)
		if weight is not None:
			weight=torch.tensor(weight.reshape(-1,1), dtype=torch.float)
		self._model=TorchNNCore(inps=X.shape[1], hiddens=self._hiddens, seed=self._seed)
		optim=torch.optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=1E-4)
		for epoch in range(self._n_epoch):
			optim.zero_grad()
			y_pred=self._model(X)
			if weight is not None:
				loss=(self._loss_func(y_pred, y) * weight).mean()
			else:
				loss=self._loss_func(y_pred, y).mean()
			loss.backward()
			optim.step()
	def advexp(self,X,y,method='FGSM',eps=0.1):
		if method=='FGSM':
			X_=torch.tensor(X, dtype=torch.float, requires_grad=True)
			y_=torch.tensor(y.reshape(-1,1), dtype=torch.float)
			y_pred=self._model(X_)
			loss=self._loss_func(y_pred,y_).mean()
			noise=eps*torch.sign(torch.autograd.grad(loss,X_)[0])
			return (X_+noise).detach().numpy()
		elif method=='PGD':
			X_=torch.tensor(X, dtype=torch.float, requires_grad=True)
			y_=torch.tensor(y.reshape(-1,1), dtype=torch.float)
			for i in range(0,10):
				y_pred=self._model(X_)
				loss=self._loss_func(y_pred,y_).mean()
				noise=(eps*0.1)*torch.sign(torch.autograd.grad(loss,X_)[0])
				X_=(X_+noise).detach().requires_grad_(True)
			return X_.detach().numpy()
	def predict(self,X):
		X_=torch.tensor(X, dtype=torch.float)
		return self._model(X_).detach().numpy().reshape(-1)
	def predict_attack(self,X,y,method='FGSM',eps=0.1):
		X_=self.advexp(X,y,method=method,eps=eps)
		X_=torch.tensor(X_, dtype=torch.float)
		return self._model(X_).detach().numpy().reshape(-1)
	def metrics(self,X,y,s=None):
		y_pred=self.predict(X)
		metric=Metric(true=y,pred=y_pred)
		if s is not None:
			acc=metric.accuracy()
			disp=metric.positive_disparity(s=s)
			return acc, disp
		else:
			return metric.accuracy()
	def metrics_attack(self,X,y,s=None,method='FGSM'):
		y_pred=self.predict_attack(X,y,method=method)
		metric=Metric(true=y,pred=y_pred)
		if s is not None:
			acc=metric.accuracy()
			disp=metric.positive_disparity(s=s)
			return acc, disp
		else:
			return metric.accuracy()

