import torch
import numpy as np
np.set_printoptions(suppress=False)
from torch.autograd import grad as gradient
from pandas import DataFrame
from matplotlib import pyplot as plt

from TorchAdversarial import TorchNNCore
from utils import load_split
from metric import Metric

class PreProcFlip(object):
	def __init__(self, data, attr, k=None, seed=None, max_iter=1000):
		self._delta = 0.01
		self._data = data
		self._attr = attr
		self._k = k
		self._seed = seed
		self._max_iter = max_iter
		self._epsilon = 0.1
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

		self._train_np, self._test_np = load_split(data, attr)
		self._train_np['y_'] = self._train_np['y'].copy()

		if self._k is None:
			self._k = np.ceil(self._train_np['X'].shape[0] * 0.003).astype(int)
		print(self._train_np['X'].shape[0])
		print('Flipping %.4f%% (%d) records every epoch'%(self._k*100.0/self._train_np['X'].shape[0], self._k))

		self._train = {
			'X':torch.tensor(self._train_np['X'], dtype=torch.float, requires_grad=True),
			'y':torch.tensor(self._train_np['y'].reshape(-1,1), dtype=torch.float, requires_grad=True),
			'y_':torch.tensor(self._train_np['y'].reshape(-1,1), dtype=torch.float),
			'c':torch.tensor(self._train_np['c'].reshape(-1,1), dtype=torch.float),
		}
		
		self._test = {
			'X':torch.tensor(self._test_np['X'], dtype=torch.float, requires_grad=True),
			'y':torch.tensor(self._test_np['y'].reshape(-1,1), dtype=torch.float),
		}


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

		model = TorchNNCore(
			inps=self._train['X'].shape[1],
			hiddens=[128],
			seed=self._seed,
			hidden_activation=torch.nn.LeakyReLU,
		)

		optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
		loss_func = torch.nn.BCELoss()

		chosen = set()

		for it in range(0, self._max_iter):

			# BEGIN: Train modified model
			tolerence = 20
			last_loss = None
			for epoch in range(0, 1000):
				optim.zero_grad()
				y_pred = model(self._train['X'])
				loss = self._BCELoss(y_pred, self._train['y_'])
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
			self._train['y'].grad = None
			self._train['X'].grad = None
			y_pred = model(self._train['X'])
			y_pred_np = y_pred.detach().numpy().reshape(-1)
			y_grad = gradient(self._BCELoss(y_pred, self._train['y']), self._train['y'])[0].reshape(-1)
			influence_to_utility = -y_grad * np.sign(self._train_np['y'] - 0.5)
			metric = Metric(true=self._train_np['y'], pred=y_pred.detach().numpy().reshape(-1))
			acc_train_org = metric.accuracy()
			disp_train_org = metric.positive_disparity(s=self._train_np['s'])
			# END: Influences of utility

			# BEGIN: Influences? of fairness
			# if disp_train_org > self._delta:
			# 	metric = Metric(pred=y_pred.detach().numpy(), true=self._train_np['y_'])
			# 	disp = metric.positive_disparity(s=self._train_np['s'], absolute=False)
			# 	influence_to_fairness = np.sign(
			# 		disp * (self._train_np['s'] - 0.5) * (self._train_np['y_'] - 0.5)
			# 	) * (0.5 - abs(y_pred_np - 0.5))
			# else:
			influence_to_fairness = np.zeros(self._train['X'].shape[0])
			# END: Influences? of fairness

			# BEGIN: Influences of robustness
			self._train['X'].grad = None
			self._train['y'].grad = None
			y_pred = model(self._train['X'])
			noise = torch.sign(gradient(self._BCELoss(y_pred, self._train['y']), self._train['X'])[0])
			y_pred_atk = model(self._train['X'] + 0.1 * noise)
			y_grad = gradient(self._BCELoss(y_pred_atk, self._train['y']), self._train['y'])[0].reshape(-1)
			influence_to_robustness = - y_grad * np.sign(self._train_np['y']-0.5)
			acc_train_atk = Metric(true=self._train_np['y'], pred=y_pred_atk.detach().numpy().reshape(-1)).accuracy()
			# END: Influences of robustness

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
				y_test_pred = model(self._test['X'])
				metric = Metric(
					true=self._test_np['y'], pred=y_test_pred.detach().numpy().reshape(-1)
				)
				acc_test_org = metric.accuracy()
				disp_test_org = metric.positive_disparity(s=self._test_np['s'])

				loss = self._BCELoss(y_test_pred, self._test['y'])
				noise = torch.sign(gradient(loss, self._test['X'])[0]) * self._epsilon
				y_test_pred_atk = model(self._test['X'] + noise)
				acc_test_atk = Metric(
					true=self._test_np['y'], pred=y_test_pred_atk.detach().numpy().reshape(-1)
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
				if i in chosen:
					continue
				if item[1] > 0 and item[2] > 0:
					break
				chosen.add(i)
				self._train['y_'][i][0] = 1 - self._train['y_'][i][0]
				self._train_np['y_'][i] = 1 - self._train_np['y_'][i]
				rest -= 1
				if rest == 0:  # Successfully picked k flippings
					break
			if rest == self._k:  # No eligible flipping is available
				break
			# END: Flipping

			if res["train"]["disp"][-1] <= self._delta:
				break

		return res


def draw(res):

	fig, ax = plt.subplots(1, 2)
	fig.set_size_inches(12.8, 4.8)

	data = np.hstack(
		[
			np.array(res["train"]["orig"]).reshape(-1, 1),
			# np.array(res["train"]["attk"]).reshape(-1, 1),
		]
	)
	df_train = DataFrame(
		data, index=res["iter"], columns=["Accuracy_Orig."]#, "Accuracy_Attk."]
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
			# np.array(res["test"]["attk"]).reshape(-1, 1),
		]
	)
	df_test = DataFrame(
		data, index=res["iter"], columns=["Accuracy_Orig."]#, "Accuracy_Attk."]
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
	model = PreProcFlip("compas", "race", k=100, max_iter=50, seed=24)
	res = model.fit_transform()
	draw(res)
	plt.show()
