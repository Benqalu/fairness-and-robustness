import tqdm
import torch
import numpy as np
from metric import Metric

class TorchNNCore(torch.nn.Module):
	def __init__(
		self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU
	):
		super(TorchNNCore, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		struct = [inps] + hiddens + [1]
		layers = []
		for i in range(1, len(struct)):
			layers.append(
				torch.nn.Linear(
					in_features=struct[i - 1], out_features=struct[i], bias=bias
				)
			)
			if i == len(struct) - 1:
				layers.append(torch.nn.Sigmoid())
			else:
				layers.append(hidden_activation())
		self.model = torch.nn.Sequential(*layers)

	def forward(self, x):
		output = self.model(x)
		return output

class TorchAdversarial(object):
	def __init__(self, lr=0.01, n_epoch=1000, method='FGSM', eps=0.1, hiddens=[], hidden_activation=torch.nn.ReLU, seed=None, l2=0.0):
		# super(TorchAdversarial, self).__init__(lr=lr, n_epoch=n_epoch, hiddens=hiddens, seed=seed)
		self._seed = seed
		self._loss_func=self._BCELoss
		self._method=method
		self._epsilon = eps
		self._hiddens = hiddens
		self._hidden_activation = hidden_activation
		self._lr = lr
		self._n_epoch = n_epoch
		self._device = torch.device('cpu')
		self._l2 = l2
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

	def _BCELoss(self, y_pred, y, reduction=False):
		if reduction:			
			return -torch.mean(
				y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred)
			)
		else:
			return -(y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred))

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
		X_adv = self._X.clone().detach().requires_grad_(True)
		for i in range(0,10):
			grad_X = torch.autograd.grad(torch.mean(self._loss_func(self._model(X_adv), self._y)), X_adv)[0]
			noise = torch.sign(grad_X).detach()
			X_adv = (X_adv + self._epsilon*0.1*noise).detach().requires_grad_(True)
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
			inps=self._X.shape[1], hiddens=self._hiddens, hidden_activation=self._hidden_activation, seed=self._seed
		)
		optim = torch.optim.Adam(
			self._model.parameters(),
			lr=self._lr, weight_decay=self._l2
		)

		tolerence = 10
		last_loss = None
		for epoch in range(0, self._n_epoch):
			optim.zero_grad()
			y_pred = self._model(self._X)
			loss = torch.mean(self._loss_func(y_pred, self._y)*self._weight)
			this_loss = loss.tolist()
			if last_loss is not None:
				if this_loss > last_loss or abs(last_loss - this_loss) < 1e-5:
					tolerence -= 1
				if tolerence == 0:
					break
			last_loss = this_loss
			loss.backward()
			optim.step()

	def AdvExp(self, X, y=None, method="FGSM", eps=0.1):
		if type(X) is torch.Tensor:
			X_ = X.clone().detach().requires_grad_(True)
		else:
			X_ = torch.tensor(X, dtype=torch.float, requires_grad=True, device=self._device)
		if method == "FGSM":
			y_pred = self._model(X_)
			if y is not None:
				y_ = torch.tensor(y.reshape(-1, 1), dtype=torch.float, device=self._device)
			else:
				y_ = torch.round(y_pred).detach()
			loss = self._loss_func(y_pred, y_).mean()
			noise = eps * torch.sign(torch.autograd.grad(loss, X_)[0])
			return (X_ + noise).detach()
		elif method == "PGD":
			if y is not None:
				y_ = torch.tensor(y.reshape(-1, 1), dtype=torch.float, device=self._device)
			else:
				y_ = None
			for i in range(0, 10):
				y_pred = self._model(X_)
				if y_ is None:
					y_ = torch.round(y_pred).detach()
				loss = self._loss_func(y_pred, y_).mean()
				noise = (eps * 0.1) * torch.sign(torch.autograd.grad(loss, X_)[0])
				X_ = (X_ + noise).detach().requires_grad_(True)
			return X_.detach()

	def predict(self, X):
		if type(X) is torch.Tensor:
			X_ = X.clone().detach()
		else:
			X_ = torch.tensor(X, dtype=torch.float, device=self._device)
		return self._model(X_).detach().cpu().numpy().reshape(-1)

	def predict_attack(self, X, y=None, method="FGSM", eps=0.1):
		X_ = self.AdvExp(X, y, method=method, eps=eps)
		return self._model(X_).detach().cpu().numpy().reshape(-1)

	def metrics(self, X, y, s=None):
		y_pred = self.predict(X)
		metric = Metric(true=y, pred=y_pred)
		if s is not None:
			acc = metric.accuracy()
			disp = metric.positive_disparity(s=s)
			return round(acc,6), round(disp,6)
		else:
			return round(metric.accuracy(),6)

	def metrics_attack(self, X, y, s=None, method="FGSM", use_y=True):
		if use_y:
			y_pred = self.predict_attack(X, y, method=method)
		else:
			y_pred = self.predict_attack(X, None, method=method)
		metric = Metric(true=y, pred=y_pred)
		if s is not None:
			acc = metric.accuracy()
			disp = metric.positive_disparity(s=s)
			return round(acc,6), round(disp,6)
		else:
			return round(metric.accuracy(),6)

if __name__=='__main__':
	from utils import load_split
	train, test = load_split('compas','race')

	model = TorchAdversarial(lr=0.01, n_epoch=500, method='PGD', hiddens=[128], seed=24)
	model.fit(train['X'], train['y'], wR=0.1)




