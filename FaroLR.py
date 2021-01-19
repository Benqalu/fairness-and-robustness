import torch
import numpy as np
from time import time
from copy import deepcopy
from metric import Metric


def loss_fairness(X, y, w, tp=False):
	if not tp:
		u_down = torch.sum(1.0 - X[:, 0])
		u_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * (1.0 - X[:, 0]))
		v_down = torch.sum(X[:, 0])
		v_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * X[:, 0])
		loss = torch.abs((u_up / u_down) - (v_up / v_down))
	else:
		y_flat=y.reshape(-1)
		u_down = torch.sum((1.0 - X[:, 0]) * y_flat)
		u_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * (1.0 - X[:, 0]) * y_flat)
		v_down = torch.sum(X[:, 0] * y_flat)
		v_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * X[:, 0] * y_flat)
		loss = torch.abs((u_up / u_down) - (v_up / v_down))
	return loss

def loss_robustness(X, y, w):
	gradx = (y.reshape(-1) - torch.sigmoid(torch.matmul(X, w))).reshape(-1, 1) * w
	loss = torch.me
	an(torch.sum(torch.abs(gradx), axis=1))
	return loss


# def loss_robustness_switch(X, y, w, eps=0.1):
# 	gradx = eps * (y.reshape(-1) - torch.sigmoid(torch.matmul(X, w))).reshape(-1, 1) * w
# 	loss = torch.mean(
# 		1.0 - torch.sigmoid(torch.matmul(X + gradx, w)) * torch.matmul(X, w)
# 	)
# 	return loss


class LogisticRegressionCore(torch.nn.Module):
	def __init__(self, input_dim, bias=True, seed=None):
		super(LogisticRegressionCore, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(input_dim, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()

	def forward(self, x):
		return self.activation(self.linear_model(x))


class FaroLR(object):
	def __init__(
		self, lr=0.01, n_epoch=1000, bias=True, fairness=0.0, robustness=0.0, tp_fairness=False, report=[], seed=None
	):
		self._lr = lr
		self._n_epoch = n_epoch
		self._model = None
		self._bias = bias
		self._fairness = fairness
		self._robustness = robustness
		self._seed = seed
		self._report = {}
		self._tp_fairness=tp_fairness
		for item in report:
			self._report[item]=[]

	def fit(self, X, y):
		X = torch.hstack([torch.tensor(X).float(), torch.ones(X.shape[0], 1)])
		X = torch.tensor(X, requires_grad=True)
		y = torch.tensor(y).reshape(-1, 1).float()

		self._model = LogisticRegressionCore(
			input_dim=X.shape[1], bias=self._bias, seed=self._seed
		)

		s = np.array(X[:, 0].tolist())

		loss_main = torch.nn.BCELoss()
		optim = torch.optim.Adam(self._model.parameters(), lr=self._lr)

		for epoch in range(0, self._n_epoch):
			optim.zero_grad()
			y_pred = self._model(X)
			loss = loss_main(y_pred, y)
			if self._fairness is not None and self._fairness != 0.0:
				loss += self._fairness * loss_fairness(
					X, y, self._model.linear_model.weight.reshape(-1), tp=self._tp_fairness
				)
			if self._robustness is not None and self._robustness != 0.0:
				# loss+=self._robustness*loss_robustness_switch(X,y,self._model.linear_model.weight.reshape(-1),eps=0.1)
				loss += self._robustness * loss_robustness(
					X, y, self._model.linear_model.weight.reshape(-1)
				)
			loss.backward()
			optim.step()

			if epoch % 10 == 0:
				metric = Metric(true=y.tolist(), pred=y_pred.tolist())
				acc = metric.accuracy()
				if not self._tp_fairness:
					disp = metric.positive_disparity(s=s)
				else:
					disp = metric.recall_disparity(s=s)
				print("epoch=%d, Acc.=%.4f, Disp.=%.4f" % (epoch, acc, disp))

				if 'weight' in self._report:
					weights = self._model.linear_model.weight.reshape(-1).tolist()
					self._report['weight'].append(weights)
				if 'weight_grad' in self._report:
					weights_grad = self._model.linear_model.weight.grad.reshape(-1).tolist()
					self._report['weight_grad'].append(weights_grad)
				if 'accuracy' in self._report or 'disparity' in self._report:
					self._report['accuracy'].append(acc)
					self._report['disparity'].append(disp)

	def predict(self, X, y=None):
		X = torch.hstack([torch.tensor(X).float(), torch.ones(X.shape[0], 1)])
		X = torch.tensor(X, requires_grad=True)
		y_pred = np.array(self._model(X).reshape(-1).tolist())
		if y is not None:
			metric = Metric(true=y.tolist(), pred=y_pred.tolist())
			return metric.accuracy()
		else:
			return y_pred

	def attack(self, X, y, eps=0.1):

		X_orig = np.array(X)

		X = torch.hstack([torch.tensor(X).float(), torch.ones(X.shape[0], 1)]).float()
		X = torch.tensor(X, requires_grad=True)
		y = torch.tensor(y).reshape(-1, 1).float()

		loss_main = torch.nn.BCELoss()
		optim = torch.optim.Adam(self._model.parameters(), lr=self._lr)

		optim.zero_grad()
		y_pred = self._model(X)
		loss = loss_main(y_pred, y)

		loss.backward()

		grads = X.grad
		noise = eps * torch.sign(grads[:, :-1])

		ret = X_orig + np.array(noise.tolist())

		return ret

	def eval_attack(self, X, y, eps=0.1):
		y_pred_acc=self.predict(X,y)
		X_attack = self.attack(X, y, eps=eps)
		y_pred_attack_acc=self.predict(X_attack, y)
		res={'origin': y_pred_acc, 'attack': y_pred_attack_acc}
		self._report['attack']=res
		return res

