import tqdm
import torch
import numpy as np
from TorchAttackable import TorchNNCore, TorchNeuralNetworks

class TorchAdversarial(TorchNeuralNetworks):
	def __init__(self, lr=0.01, n_epoch=1000, method='FGSM', eps=0.1, hiddens=[], seed=None):
		super(TorchAdversarial, self).__init__(lr=lr, n_epoch=n_epoch, hiddens=hiddens, seed=seed)
		self._seed = seed
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)
		self._loss_func=torch.nn.BCELoss(reduction='none')
		self._method=method
		self._epsilon = eps

	def loss_robustness_fgsm(self, paritial=True):
		noise = torch.sign(self._X.grad).detach()
		if not paritial:
			X_adv = (self._X + self._epsilon * noise).detach()
			y_adv_pred = self._model(X_adv)
			return torch.mean(self._loss_func(y_adv_pred, self._y))
		else:
			X_adv = (self._X[self._idx] + self._epsilon * noise[self._idx]).detach()
			y_adv_pred = self._model(X_adv)
			return torch.mean(self._loss_func(y_adv_pred, self._y[self._idx]))

	def loss_robustness_pgd(self, paritial=True):
		noise = torch.sign(self._X.grad).detach()
		X_adv = self._X.clone().detach().requires_grad_(True)
		for i in range(0,10):
			X_adv = (X_adv + self._epsilon*0.1*noise).detach().requires_grad_(True)
			torch.mean(self._loss_func(self._model(X_adv), self._y)*self._weight).backward()
			noise = torch.sign(X_adv).detach()
		X_adv.detach_()
		if not paritial:
			y_adv_pred = self._model(X_adv)
			return torch.mean(self._loss_func(y_adv_pred, self._y))
		else:
			y_adv_pred = self._model(X_adv[self._idx])
			return torch.mean(self._loss_func(y_adv_pred, self._y[self._idx]))

	def fit(self, X, y, s=None, weight=None, wR=0.0):

		self._idx=np.random.choice(np.arange(0,X.shape[0]), int(X.shape[0] * wR * 0.1), replace=False)

		self._X = torch.tensor(X, dtype=torch.float, requires_grad=True)
		self._y = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
		if weight is not None:
			self._weight = torch.tensor(weight.reshape(-1, 1), dtype=torch.float)
		else:
			self._weight = torch.ones((X.shape[0],1))
		self._s=s

		self._model = TorchNNCore(
			inps=self._X.shape[1], hiddens=self._hiddens, seed=self._seed
		)
		optim = torch.optim.Adam(
			self._model.parameters(),
			lr=self._lr,  # weight_decay=wR
		)

		for epoch in tqdm.tqdm(range(self._n_epoch)):
			y_pred = self._model(self._X)

			loss_u = torch.mean(self._loss_func(y_pred, self._y)*self._weight)
			if self._X.grad is None:
				loss_r = torch.tensor(0.,dtype=torch.float)
			else:
				if self._method=='FGSM' and wR>0.0:
					loss_r = self.loss_robustness_fgsm(paritial=True)
				elif self._method=='PGD' and wR>0.0:
					loss_r = self.loss_robustness_pgd(paritial=True)
				else:
					loss_r = torch.tensor(0.,dtype=torch.float)
				self._X.grad = None

			loss = loss_u+wR*loss_r

			optim.zero_grad()
			loss.backward()
			optim.step()

if __name__=='__main__':
	from utils import load_split
	train, test = load_split('compas','race')

	model = TorchAdversarial(lr=0.01, n_epoch=500, method='PGD', hiddens=[128], seed=24)
	model.fit(train['X'], train['y'], wR=0.1)




