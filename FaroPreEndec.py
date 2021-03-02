import torch
import numpy as np

from utils import get_data
from metric import Metric

from sklearn.linear_model import LogisticRegression as SKLogisticRegression

class TorchLogistic(torch.nn.Module):
	def __init__(self, inps, bias=True, seed=None):
		super(TorchLogistic, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(inps, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()
	def forward(self, x):
		return self.activation(self.linear_model(x))

class TorchAutoCodec(torch.nn.Module):
	def __init__(self, hiddens, inouts, hidden_activation=torch.nn.LeakyReLU):
		super().__init__()
		struct = [inouts]+hiddens+[inouts]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i],bias=True))
			layers.append(hidden_activation())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		return output
		

class FaroAE(object):

	def __init__(self, n_epoch=3000, lr=0.01, wf=0.0, wr=0.0, hiddens=[128]):
		self._wf=wf
		self._wr=wr
		self._learning_rate=lr
		self._n_epoch=n_epoch
		self._inouts=None
		self._hiddens=hiddens
		self._model_codec=None
		self._model_target=None
		self._c=None
		self._bce_loss=torch.nn.BCELoss()

	def _loss_utility(self,X,outs):
		return torch.mean(torch.abs(X-outs))

	def _loss_fairness(self,X):
		y_=self._model_target(X)
		return torch.abs(torch.sum(self._c*y_))

	def _loss_robustness(self):
		w=[item for item in self._model_target.parameters()]
		return torch.mean(torch.abs(w))

	def fit_transform(self,X,y):

		self._np_s=X[:,0].reshape(-1)
		self._np_X=X[:,1:]
		self._np_y=y.reshape(-1)

		s = torch.tensor(self._np_s.reshape(-1,1), dtype=torch.float, requires_grad=False)
		X = torch.tensor(self._np_X, dtype=torch.float, requires_grad=False)
		y = torch.tensor(self._np_y.reshape(-1,1), dtype=torch.float, requires_grad=False)
		self._inouts=X.shape[1]
		self._c = (1.0-s-s)/((1-s)*torch.sum(1-s)+s*torch.sum(s))
		
		self._model_codec=TorchAutoCodec(hiddens=self._hiddens, inouts=self._inouts)
		self._model_target=TorchLogistic(inps=self._inouts)
		optim_codec = torch.optim.Adam(
			self._model_codec.parameters(), lr=self._learning_rate, weight_decay=0.001
		)
		optim_target = torch.optim.Adam(
			self._model_target.parameters(), lr=self._learning_rate
		)

		for epoch in range(0, self._n_epoch):
			optim_target.zero_grad()
			outs=self._model_target(X)
			loss=self._bce_loss(outs,y)
			loss.backward()
			optim_target.step()
		print(loss.tolist())

		test_model=SKLogisticRegression(max_iter=1000)

		for it in range(0, self._n_epoch):
			optim_codec.zero_grad()
			X_=self._model_codec(X)
			lossU=self._loss_utility(X=X,outs=X_)
			loss=lossU
			if self._wr>0:
				lossR=self._loss_robustness()
				loss+= self._wr * lossR
			if self._wf>0:
				lossF=self._loss_fairness(X_)
				loss+= self._wf * lossF
			loss.backward()
			optim_codec.step()

			if it%10==0:
				print('Codec: %d %.6f %.6f'%(it, lossU.tolist(), lossF.tolist()),end='\t')
				test_model.fit(X_.tolist(),self._np_y)
				y_pred=test_model.predict(self._np_X)
				metric=Metric(pred=y_pred,true=self._np_y)
				print('%.4f %.4f'%(metric.accuracy(), metric.positive_disparity(s=self._np_s)))

			for epoch in range(0,1):
				optim_target.zero_grad()
				outs=self._model_target(self._model_codec(X))
				lossT=self._bce_loss(outs,y)
				lossT.backward()
				optim_target.step()

		return torch.hstack([s.reshape(-1,1),X]).detach().numpy(), y


if __name__=='__main__':
	model = FaroAE(
		hiddens=[32,32], 
		n_epoch=5000, 
		lr=0.001,
		wf=0.01
	)
	X,y=get_data('compas','sex')
	X_,y_=model.fit_transform(X,y)

	from sklearn.linear_model import LogisticRegression
	validation=LogisticRegression()
	validation.fit(X_,y_.reshape(-1))
	acc=validation.score(X[:2000],y[:2000])
	print(acc)

	validation=LogisticRegression()
	validation.fit(X[2000:4000],y[2000:4000].reshape(-1))
	acc=validation.score(X[:2000],y[:2000])
	print(acc)




