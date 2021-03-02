import sys
import tqdm
import torch
import numpy as np

from metric import Metric
from utils import get_data

class TorchLogistic(torch.nn.Module):
	def __init__(self, inps, bias=True, seed=None):
		super(TorchLogistic, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(inps, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()
	def forward(self, x):
		return self.activation(self.linear_model(x))

class FaroPre(object):

	def __init__(self, n_epoch=3000, n_iter=1000, lr=0.001, wf=0.0, wr=0.0):
		self._wf=wf
		self._wr=wr
		self._learning_rate=lr
		self._n_epoch=n_epoch
		self._n_iter=n_iter
		self._inps=None
		self._utility_model=None
		self._bce_loss=torch.nn.BCELoss()

	def _loss_utility(self,X,y,s=None):
		# return torch.mean(torch.abs(X-self._original_X))
		outs=self._utility_model(X)
		loss=self._bce_loss(outs,y)
		return loss

	def _loss_fairness(self,X,y=None,s=None):
		outs=self._model(X)
		return torch.abs(torch.sum(self._c*outs))

	def _loss_robustness(self,X=None,y=None,s=None):
		w=[item for item in self._model.parameters()]
		return torch.mean(torch.abs(w))

	def _train_utility_model(self,X,y):

		print('Training utility model:')
		self._utility_model=TorchLogistic(inps=self._inps)
		optim = torch.optim.Adam(
			self._utility_model.parameters(), lr=self._learning_rate
		)
		loss_func=self._bce_loss
		for epoch in range(1, self._n_epoch + 1):
			optim.zero_grad()
			outs=self._utility_model(X)
			loss=loss_func(outs,y)
			loss.backward()
			optim.step()
			if epoch%10==0:
				print('Epoch: %d / %d, loss=%.6f'%(epoch, self._n_epoch, loss.tolist()),end='\r')
				sys.stdout.flush()
		print()
		metric=Metric(true=y.numpy(), pred=outs.detach().numpy())
		print('Training accuracy: %.4f'%metric.accuracy())

	def fit_transform(self,X,y):
		s = torch.tensor(X[:, 0].reshape(-1,1), requires_grad=False)
		y = torch.tensor(y, dtype=torch.float, requires_grad=False)
		X = torch.tensor(X[:, 1:], dtype=torch.float, requires_grad=True)
		self._inps=X.shape[1]

		self._original_X=X.clone()
		self._original_y=y.clone()
		self._original_s=s.clone()

		# self._train_utility_model(X,y)

		self._model=TorchLogistic(inps=self._inps)
		self._c = (1.0-s-s)/((1-s)*torch.sum(1-s)+s*torch.sum(s))
		optM = torch.optim.Adam(
			self._model.parameters(), lr=self._learning_rate
		)
		optX = torch.optim.Adam(
			[X], lr=self._learning_rate * 0.1
		)
		loss_func=self._bce_loss

		last_X=np.array(X.tolist())
		orig_X=np.array(self._original_X.tolist())
		initial_diff=None
		for epoch in range(0, self._n_iter):

			backward_n=0
			last_loss=None
			for i in range(0,self._n_epoch):
				optM.zero_grad()
				outs=self._model(X)
				loss=loss_func(outs,y)
				loss.backward()
				optM.step()
				
				# this_loss=loss.tolist()
				# if last_loss is None:
				# 	last_loss=loss.tolist()
				# elif loss.tolist()>last_loss:
				# 	backward_n+=1
				# 	if backward_n>=3:
				# 		break
				# last_loss=this_loss

			optX.zero_grad()

			lossU=self._loss_utility(X,y)
			loss=lossU
			if self._wf>0:
				lossF=self._loss_fairness(X)
				loss+=self._wf*lossF
			if self._wr>0:
				loss+=self._wr*lossR
				lossR+=self._loss_robustness()

			loss.backward()
			optX.step()

			# this_X=np.array(X.tolist())
			# diff=np.max(abs(this_X-last_X))
			# total_diff=np.mean(abs(this_X-orig_X))
			# last_X=this_X

			# print(lossF.tolist())
			# print('Iter: %d / %d, diff: %.6f, total: %.6f'%(epoch+1, self._n_iter, diff, total_diff),end='\r')
			# sys.stdout.flush()

			# if initial_diff is None:
			# 	initial_diff=diff
			# elif diff<initial_diff*0.05:
			# 	break

			print(lossU.tolist(), lossF.tolist())

		# print()

		return X.tolist()


if __name__=='__main__':
	model = FaroPre(
		n_epoch=3000, 
		lr=0.01,
		wf=0.50,
		wr=0.00,
	)
	data='adult'
	attr='race'
	X,y=get_data(data,attr)
	X=model.fit_transform(X,y)
	f=open('transformed_X_%s_%s.txt'%(data,attr),'w')
	f.write(str(X))
	f.close()

