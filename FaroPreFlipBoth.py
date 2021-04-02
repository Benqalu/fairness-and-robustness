import torch
import time
import tqdm
import random
import numpy as np

np.set_printoptions(suppress=False)
from torch.autograd import grad as gradient
from sklearn.model_selection import train_test_split

from TorchAttackable import TorchNNCore

from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from pandas import DataFrame
from metric import Metric
from utils import get_data

from matplotlib import pyplot as plt

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print("Seed is %d" % seed)


class TorchLogistic(torch.nn.Module):
	def __init__(self, inps, bias=True, seed=None):
		super(TorchLogistic, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(inps, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()

	def forward(self, x):
		return self.activation(self.linear_model(x))


class PreProcFlip(object):
	def __init__(self, data, attr, k=10, seed=24, credit_lim=1, max_iter=1000):
		self._delta = 0.01
		self._data = data
		self._attr = attr
		self._k = k
		self._seed = seed
		self._credit_limit = credit_lim
		self._max_iter = max_iter
		self._epsilon = 0.1
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

		X, y = get_data(data, attr)
		self._orig_s = X[:, 0].reshape(-1)
		self._orig_X = X[:, 1:]
		self._orig_y = y.reshape(-1)

		X, X_test, s, s_test, y, y_test = train_test_split(
			self._orig_X, self._orig_s, self._orig_y, test_size=0.3
		)

		self._train_pack_np = (X, s, y, y.copy())
		self._test_pack_np = (X_test, s_test, y_test)

		X_prime = torch.tensor(X, dtype=torch.float)
		y_prime = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
		X = torch.tensor(X, requires_grad=True, dtype=torch.float)
		s = torch.tensor(s.reshape(-1, 1), dtype=torch.float)
		y = torch.tensor(y.reshape(-1, 1), requires_grad=True, dtype=torch.float)

		X_test = torch.tensor(X_test, requires_grad=True, dtype=torch.float)
		y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)

		self._train_pack = (X, s, y, X_prime, y_prime)
		self._test_pack = (X_test, s_test, y_test)

		if self._k is None:
			self._k = np.ceil(X.shape[0] * 0.001).astype(int)

	def _BCELoss(self, y_pred, y):
		return -torch.mean(
			y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred)
		)

	def _DISPLoss(self, c, y_pred):
		return torch.square(torch.sum(c * y_pred))

	def _AccDisp(self, y, y_pred, s):
		metric = Metric(true=y.reshape(-1).tolist(), pred=y_pred.reshape(-1).tolist())
		return metric.accuracy(), metric.positive_disparity(s=s)

	def fit_transform(self, test_output=True):

		X, s, y, X_prime, y_prime = self._train_pack
		X_test, s_test, y_test = self._test_pack

		X_np, s_np, y_np, y_np_prime = self._train_pack_np
		X_test_np, s_test_np, y_test_np = self._test_pack_np

		c_train = (1 - s_np - s_np) / (
			(1 - s_np) * np.sum(1 - s_np) + s_np * np.sum(s_np)
		)
		c_train = torch.tensor(c_train, dtype=torch.float)

		res = {
			"setting": "%s_%s" % (self._data, self._attr),
			"train": {
				"orig": [],
				"attk": [],
				"disp": [],
			},
			"test": {
				"orig": [],
				"attk": [],
				"disp": [],
			},
			"iter": [],
		}

		model = TorchNNCore(inps=X.shape[1], hiddens=[128], seed=self._seed)
		optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
		loss_func = torch.nn.BCELoss()
		credit = {}

		for it in range(0, self._max_iter):

			# BEGIN: Train modified model
			tolerence = 20
			last_loss = None
			for epoch in range(0, 500):
				optim.zero_grad()
				y_pred = model(X)
				loss = self._BCELoss(y_pred, y_prime)
				this_loss = loss.tolist()
				if last_loss is not None:
					if this_loss > last_loss or abs(last_loss - this_loss) < 1e-5:
						tolerence -= 1
					if tolerence == 0:
						break
				last_loss = this_loss
				loss.backward()
				optim.step()
			# END

			res["iter"].append(it + 1)

			# BEGIN: Influences of utility
			y.grad = None
			X.grad = None
			y_pred = model(X)
			loss = self._BCELoss(y_pred, y)
			loss.backward()
			y_grad = y.grad.numpy().reshape(-1)
			influence_to_utility = -y_grad * np.sign(y_np - 0.5)  # Condtion 1: Utility
			metric = Metric(true=y_np, pred=y_pred.detach().numpy().reshape(-1))
			acc_train_org = metric.accuracy()
			disp_train_org = metric.positive_disparity(s=s_np)
			# END: Influences of utility

			# BEGIN: Influences? of fairness
			# if disp_train_org > self._delta:
			# 	metric = Metric(pred=y_pred.detach().numpy(), true=y_np_prime)
			# 	disp = metric.positive_disparity(s=s_np, absolute=False)
			# 	influence_to_fairness = np.sign(
			# 		disp * (s_np - 0.5) * (y_np_prime - 0.5)
			# 	)
			# else:
			# 	influence_to_fairness = np.zeros(X.shape[0])
			# END: Influences? of fairness

			# BEGIN: Influences of fairness
			y.grad = None
			X.grad = None
			y_pred = model(X)

			loss_f = self._DISPLoss(c_train, y_pred)
			grad_w_F = gradient(loss_f, model.parameters(), retain_graph=True)
			grad_w_F = torch.hstack([torch.flatten(item) for item in grad_w_F])

			loss_u = self._BCELoss(y_pred, y)
			grad_w_U = gradient(loss_u, model.parameters(), retain_graph=True, create_graph=True)
			grad_w_U = torch.hstack([torch.flatten(item) for item in grad_w_U])
			H = []
			s_time=time.time()
			for i in tqdm.tqdm(range(0,grad_w_U.shape[0])):
				grad_w2_U = gradient(grad_w_U[i], model.parameters(), retain_graph=True)
				grad_w2_U = torch.hstack([torch.flatten(item) for item in grad_w2_U])
				H.append(grad_w2_U.tolist())
			H = np.array(H)
			exit()

			params = torch.hstack([torch.flatten(item) for item in model.parameters()])

			grad_u = gradient(loss, params, retain_graph=True, create_graph=True)
			grad_u_y = gradient(grad_u, y, )
			# END: Influences? of fairness

			# BEGIN: Influences of robustness
			# noise = (torch.sign(X.grad) * self._epsilon).clone().detach()
			# X.grad = None
			# y.grad = None
			# y_pred_atk = model(X+noise)
			# loss = self._BCELoss(y_pred_atk, y)
			# loss.backward()
			# y_grad = y.grad.numpy().reshape(-1)
			# influence_to_robustness = - y_grad * np.sign(y_np-0.5)
			# acc_train_atk = Metric(true=y_np, pred=y_pred_atk.detach().numpy().reshape(-1)).accuracy()
			acc_train_atk = -1.0
			influence_to_robustness = np.zeros(influence_to_utility.shape[0])
			# END: Influences

			print(
				"Iter: %d, Train: (%.4f, %.4f, %.4f)"
				% (it + 1, acc_train_org, acc_train_atk, disp_train_org),
				end=", ",
			)
			res["train"]["orig"].append(acc_train_org)
			res["train"]["attk"].append(acc_train_atk)
			res["train"]["disp"].append(disp_train_org)

			# BEGIN: Testing
			if test_output:
				# X_test.grad=None
				y_test_pred = model(X_test)
				metric = Metric(
					true=y_test_np, pred=y_test_pred.detach().numpy().reshape(-1)
				)
				acc_test_org = metric.accuracy()
				disp_test_org = metric.positive_disparity(s=s_test_np)

				loss = self._BCELoss(y_test_pred, y_test)
				noise = torch.sign(gradient(loss, X_test)[0]) * self._epsilon
				y_test_pred_atk = model(X_test + noise)
				acc_test_atk = Metric(
					true=y_test_np, pred=y_test_pred_atk.detach().numpy().reshape(-1)
				).accuracy()

				print(
					"Test: (%.4f, %.4f, %.4f)"
					% (acc_test_org, acc_test_atk, disp_test_org)
				)
				res["test"]["orig"].append(acc_test_org)
				res["test"]["attk"].append(acc_test_atk)
				res["test"]["disp"].append(disp_test_org)
			# END: Testing

			# BEGIN: Choosing Strategy
			influence = np.column_stack(
				(
					influence_to_fairness,
					influence_to_robustness,
					influence_to_utility,
					list(range(0, influence_to_utility.shape[0])),
				)
			)
			influence = influence.tolist()
			influence.sort()
			# END: Choosing Strategy

			# BEGIN: Flipping
			rest = self._k
			for item in influence:
				i = int(item[3])
				if i not in credit:
					credit[i] = self._credit_limit
				elif credit[i] == 0:
					continue
				if item[1] > 0 and item[2] > 0:
					break
				credit[i] -= 1
				y_prime[i][0] = 1 - y_prime[i][0]
				y_np_prime[i] = 1 - y_np_prime[i]
				rest -= 1
				if rest == 0:  # Successfully picked k flippings
					break
			if rest == self._k:  # No eligible flipping is available
				break
			# END: Flipping

		return res


def draw(res):

	fig, ax = plt.subplots(1, 2)
	fig.set_size_inches(12.8, 4.8)

	data = np.hstack(
		[
			np.array(res["train"]["orig"]).reshape(-1, 1),
			np.array(res["train"]["attk"]).reshape(-1, 1),
		]
	)
	df_train = DataFrame(
		data, index=res["iter"], columns=["Accuracy_Orig.", "Accuracy_Attk."]
	)
	main_ax = df_train.plot(ax=ax[0])
	main_ax.set_xlabel("Training epochs")
	main_ax.set_ylabel("Accuracy")
	main_ax.set_title(res["setting"] + "_train")

	data = np.array(res["train"]["disp"]).reshape(-1, 1)
	df_train = DataFrame(data, index=res["iter"], columns=["Disparity"])
	m2nd_ax = df_train.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel("Statistical Parity")

	data = np.hstack(
		[
			np.array(res["test"]["orig"]).reshape(-1, 1),
			np.array(res["test"]["attk"]).reshape(-1, 1),
		]
	)
	df_test = DataFrame(
		data, index=res["iter"], columns=["Accuracy_Orig.", "Accuracy_Attk."]
	)
	main_ax = df_test.plot(ax=ax[1])
	main_ax.set_xlabel("Training epochs")
	main_ax.set_ylabel("Accuracy")
	main_ax.set_title(res["setting"] + "_test")

	data = np.array(res["test"]["disp"]).reshape(-1, 1)
	df_test = DataFrame(data, index=res["iter"], columns=["Disparity"])
	m2nd_ax = df_test.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel("Statistical Parity")

	fig.tight_layout()


if __name__ == "__main__":
	model = PreProcFlip("adult", "race", max_iter=100, k=10, credit_lim=3)
	res = model.fit_transform()
	draw(res)
	plt.show()
