import torch
import numpy as np
from time import time
from copy import deepcopy
from metric import Metric


def loss_fairness(X, s, y, w, tp=False):
	if not tp:
		u_down = torch.sum(1.0 - s)
		u_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * (1.0 - s))
		v_down = torch.sum(s)
		v_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * s)
	else:
		y_flat=y.reshape(-1)
		u_down = torch.sum((1.0 - s) * y_flat)
		u_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * (1.0 - s) * y_flat)
		v_down = torch.sum(s * y_flat)
		v_up = torch.sum(torch.sigmoid(torch.matmul(X, w)) * s * y_flat)
	p0=u_up/u_down
	p1=v_up/v_down
	loss = torch.abs(p0 - p1)
	return loss

def loss_robustness(X, y, w):
	# y_reshaped = y.reshape(-1)
	# err = y_reshaped*(1.0-torch.sigmoid(torch.matmul(X, w))) + (1-y_reshaped)*torch.sigmoid(torch.matmul(X, w))
	err=torch.nn.BCELoss()(torch.sigmoid(torch.matmul(X, w)).reshape(-1,1), y)
	loss = err * torch.sum(torch.square(w))
	# loss = torch.sum(torch.square(w))
	return loss


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
		self, lr=0.01, n_epoch=1000, bias=True, fairness=0.0, robustness=0.0, tp_fairness=False, report=[], comp=False, seed=None
	):
		self._lr = lr
		self._n_epoch = n_epoch
		self._model = None
		self._bias = bias
		self._fairness = fairness
		self._robustness = robustness
		self._seed = seed
		np.random.seed(self._seed)
		torch.manual_seed(self._seed)
		self._report = {}
		self._tp_fairness=tp_fairness
		self._compromised=comp
		for item in report:
			self._report[item]=[]
		self._report['epoch']=[]
		self._include_s=None

	def fit(self, X, y, X_test=None, y_test=None, include_s=True):

		print('>>> Include s:', include_s)
		self._include_s=include_s

		s = torch.tensor(X[:, 0], dtype=torch.float)
		self._report['s']=s.reshape(-1).tolist()
		if self._include_s:
			X = torch.tensor(X, dtype=torch.float)
		else:
			X = torch.tensor(X[:, 1:], dtype=torch.float)
		y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)

		if X_test is not None and y_test is not None:
			s_test = torch.tensor(X_test[:, 0], dtype=torch.float)
			self._report['s_test']=s_test.reshape(-1).tolist()
			if self._include_s:
				X_test = torch.tensor(X_test, dtype=torch.float)
			else:
				X_test = torch.tensor(X_test[:, 1:], dtype=torch.float)
			y_test = torch.tensor(y_test, dtype=torch.float).reshape(-1, 1)

		self._model = LogisticRegressionCore(
			input_dim=X.shape[1], bias=self._bias, seed=self._seed
		)

		loss_main = torch.nn.BCELoss()
		optim = torch.optim.Adam(self._model.parameters(), lr=self._lr)

		for epoch in range(0, self._n_epoch):

			optim.zero_grad()
			y_pred = self._model(X)
			loss = loss_main(y_pred, y)

			f_loss = self._fairness * loss_fairness(
				X, s, y, self._model.linear_model.weight.reshape(-1), tp=self._tp_fairness
			)

			r_loss = self._robustness * loss_robustness(
				X, y, self._model.linear_model.weight.reshape(-1)
			)

			_loss = loss
			if self._fairness is not None and self._fairness != 0.0:
				_loss += f_loss
			if self._robustness is not None and self._robustness != 0.0:
				_loss += r_loss

			_loss.backward()
			optim.step()

			if epoch % 1 == 0:
				metric = Metric(true=y.tolist(), pred=y_pred.tolist())
				acc = metric.accuracy()
				if not self._tp_fairness:
					disp = metric.positive_disparity(s=s.numpy())
				else:
					disp = metric.truepos_disparity(s=s.numpy())
				print("epoch=%d, Acc.=%.4f, Disp.=%.4f" % (epoch, acc, disp))
				self._report['epoch'].append(epoch+1)
				if 'weight' in self._report:
					weights = self._model.linear_model.weight.reshape(-1).tolist()
					self._report['weight'].append(weights)
				if 'weight_grad' in self._report:
					weights_grad = self._model.linear_model.weight.grad.reshape(-1).tolist()
					self._report['weight_grad'].append(weights_grad)
				if 'accuracy' in self._report or 'disparity' in self._report:
					self._report['accuracy'].append(acc)
					self._report['disparity'].append(disp)
					if X_test is not None and y_test is not None:
						if 'accuracy_test' not in self._report:
							self._report['accuracy_test']=[]
							self._report['disparity_test']=[]
						y_test_pred=self._model(X_test)
						metric_test = Metric(true=y_test.tolist(), pred=y_test_pred.tolist())
						acc_test = metric_test.accuracy()
						if not self._tp_fairness:
							disp_test = metric_test.positive_disparity(s=s_test.numpy())
						else:
							disp_test = metric_test.recall_disparity(s=s_test.numpy())
						self._report['accuracy_test'].append(acc_test)
						self._report['disparity_test'].append(disp_test)
				if 'loss_utility' in self._report:
					self._report['loss_utility'].append(loss.tolist())
				if 'loss_fairness' in self._report:
					self._report['loss_fairness'].append(f_loss.tolist())
				if 'loss_robustness' in self._report:
					self._report['loss_robustness'].append(r_loss.tolist())

		self._report['y_true']=y.reshape(-1).tolist()
		self._report['y_pred']=self._model(X).reshape(-1).tolist()
		X_attack = self.attack(X,y)
		y_pred_attack = self._model(X_attack).reshape(-1).tolist()
		self._report['y_pred_attack']=y_pred_attack
		metric = Metric(true=self._report['y_true'], pred=self._report['y_pred_attack'])
		if not self._tp_fairness:
			self._report['attack']=(metric.accuracy(), metric.positive_disparity(s=np.array(self._report['s']), absolute=False))
		else:
			self._report['attack']=(metric.accuracy(), metric.truepos_disparity(s=np.array(self._report['s']), absolute=False))
		
		if X_test is not None and y_test is not None:
			self._report['y_test_true']=y_test.reshape(-1).tolist()
			self._report['y_test_pred']=self._model(X_test).reshape(-1).tolist()
			X_test_attack = self.attack(X_test,y_test)
			y_test_pred_attack = self._model(X_test_attack).reshape(-1).tolist()
			self._report['y_test_pred_attack']=y_test_pred_attack
			metric = Metric(true=self._report['y_test_true'], pred=self._report['y_test_pred_attack'])
			if not self._tp_fairness:
				self._report['attack_test']=(metric.accuracy(), metric.positive_disparity(s=np.array(self._report['s_test']), absolute=False))
			else:
				self._report['attack_test']=(metric.accuracy(), metric.truepos_disparity(s=np.array(self._report['s_test']), absolute=False))

		return self._report

	def predict(self, X, y=None):
		y_pred = self._model(X).reshape(-1).tolist()
		if y is not None:
			metric = Metric(true=y.tolist(), pred=y_pred.tolist())
			return metric.accuracy(), y_pred
		else:
			return y_pred

	def attack(self, X, y, eps=0.1):

		X_orig = X.numpy()
		X = torch.tensor(X, requires_grad=True, dtype=torch.float)
		y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)

		loss_main = torch.nn.BCELoss()

		X.grad=None
		y_pred = self._model(X)
		loss = loss_main(y_pred, y)
		loss.backward()

		grads = X.grad
		noise = eps * torch.sign(grads)

		ret = torch.tensor(X_orig + np.array(noise.tolist()), dtype=torch.float)

		return ret

	def eval_attack(self, X, y, X_test=None, y_test=None, eps=0.1):
		if not self._include_s:
			X = torch.tensor(X[:, 1:], dtype=torch.float)
		y = torch.tensor(y, dtype=torch.float)
		y_pred_acc, y_pred=self.predict(X, y)
		X_attack = self.attack(X, y, eps=eps)
		y_pred_attack_acc, y_pred_attack=self.predict(X_attack, y)
		res={'origin': y_pred_acc, 'attack': y_pred_attack_acc, 'origin_pred':y_pred.tolist(), 'attack_pred':y_pred_attack.tolist()}
		return res

