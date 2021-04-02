import torch
import numpy as np
from copy import deepcopy
from torch.autograd import grad as gradient

from utils import load_split
from metric import Metric
from TorchAttackable import TorchNNCore, TorchNeuralNetworks


class PreProcess(TorchNeuralNetworks):
	def __init__(self, data, attr, seed=None, k=10, n_iter=1000):
		self._delta = 0.01
		self._data = data
		self._attr = attr
		self._k = k
		self._n_iter = n_iter
		self._seed=seed
		if self._seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
		self._train_np, self._test_np = load_split(data, attr)
		self._train_np["y_"] = deepcopy(self._train_np["y"])
		self._train = {
			"X": torch.tensor(self._train_np["X"], dtype=torch.float, requires_grad=True),
			"y": torch.tensor(
				self._train_np["y"].reshape(-1, 1), dtype=torch.float, requires_grad=True
			),
			"y_": torch.tensor(
				self._train_np["y"].reshape(-1, 1), dtype=torch.float, requires_grad=True
			),
			"s": torch.tensor(self._train_np["s"].reshape(-1, 1), dtype=torch.float),
			"c": torch.tensor(self._train_np["c"].reshape(-1, 1), dtype=torch.float),
		}
		self._test = {
			"X": torch.tensor(self._test_np["X"], dtype=torch.float),
			"y": torch.tensor(self._test_np["y"].reshape(-1, 1), dtype=torch.float),
			"s": torch.tensor(self._test_np["s"].reshape(-1, 1), dtype=torch.float),
		}

	def _BCELoss(self, y_pred, y):
		return -torch.mean(
			y * torch.log(0.999 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.999 * y_pred)
		)

	def _DISPLoss(self, c, y_pred):
		return torch.abs(torch.sum(c * y_pred))

	def _AccDisp(self, y, y_pred, s):
		metric = Metric(true=y.reshape(-1).tolist(), pred=y_pred.reshape(-1).tolist())
		return metric.accuracy(), metric.positive_disparity(s=s)

	def fit_transform(self, testing=True):

		res = {
			"setting": "%s_%s" % (self._data, self._attr),
			"train": {
				"acc": [],
				"disp": [],
			},
			"test": {
				"acc": [],
				"disp": [],
			},
			"iter": [],
		}

		self._model = TorchNNCore(inps=self._train['X'].shape[1], hiddens=[128], seed=self._seed)
		optim = torch.optim.Adam(self._model.parameters(), lr=0.01)

		chosen = set()

		for it in range(0, self._n_iter):

			# BEGIN: Train modified model
			last_loss = None
			backward_remains = 20
			for epoch in range(0, 500):
				optim.zero_grad()
				y_pred =self._model(self._train["X"])
				loss = self._BCELoss(y_pred, self._train["y_"])
				this_loss = loss.tolist()
				if last_loss is not None:
					if last_loss < this_loss:
						backward_remains -= 1
						if backward_remains == 0:
							break
				last_loss = this_loss
				loss.backward()
				optim.step()

			acc, disp = self._AccDisp(self._train["y_"], y_pred, self._train_np["s"])
			if disp <= self._delta:
				break
			res["train"]["acc"].append(acc)
			res["train"]["disp"].append(disp)
			if testing:
				print("Iter: %d, Train: (%.4f, %.4f)" % (it + 1, acc, disp), end=", ")
			else:
				print("Iter: %d, Train: (%.4f, %.4f)" % (it + 1, acc, disp))
			# END

			# BEGIN: Testing
			if testing:
				y_test_pred = self._model(self._test['X'])
				acc, disp = self._AccDisp(
					self._test["y"], y_test_pred, self._test_np["s"]
				)
				res["test"]["acc"].append(acc)
				res["test"]["disp"].append(disp)
				print("Test: (%.4f, %.4f)" % (acc, disp))
			# END: Testing

			res["iter"].append(it + 1)

			# BEGIN: Influences
			# Condtion 1: Utility
			self._train['y'].grad = None
			y_pred = self._model(self._train['X'])
			loss = self._BCELoss(y_pred, self._train['y'])
			y_pred = y_pred.detach().numpy()
			y_grad = gradient(loss, self._train['y'])[0].numpy().reshape(-1)
			influence_to_utility = -y_grad * np.sign(self._train_np['y'] - 0.5)  
			# Condition 2: Fairness
			metric = Metric(pred=y_pred, true=self._train_np['y_'])
			disp = metric.positive_disparity(s=self._train_np['s'], absolute=False)
			influence_to_parity = np.sign(disp * (self._train_np['s'] - 0.5) * (self._train_np['y_'] - 0.5))
			# END: Influences

			# BEGIN: Flipping
			influence = np.column_stack(
				(
					influence_to_parity,
					influence_to_utility,
					list(range(0, influence_to_utility.shape[0])),
				)
			)
			influence = influence.tolist()
			influence.sort()

			rest = self._k
			for item in influence:
				if item[0] > 0:
					break
				i = int(item[2])
				if i in chosen:
					continue
				chosen.add(i)
				self._train['y_'][i][0] = 1 - self._train['y_'][i][0]
				self._train_np['y_'][i] = 1 - self._train_np['y_'][i]
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

	data = np.array(res["train"]["acc"]).reshape(-1, 1)
	df_train = DataFrame(data, index=res["iter"], columns=["Train Acc."])
	main_ax = df_train.plot(ax=ax[0])
	main_ax.set_xlabel("Training epochs")
	main_ax.set_ylabel("Accuracy")
	main_ax.set_title(res["setting"] + "_train")

	data = np.array(res["train"]["disp"]).reshape(-1, 1)
	df_train = DataFrame(data, index=res["iter"], columns=["Train Disp."])
	m2nd_ax = df_train.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel("Statistical parity")

	data = np.array(res["test"]["acc"]).reshape(-1, 1)
	df_train = DataFrame(data, index=res["iter"], columns=["Test Acc."])
	main_ax = df_train.plot(ax=ax[1])
	main_ax.set_xlabel("Training epochs")
	main_ax.set_ylabel("Accuracy")
	main_ax.set_title(res["setting"] + "_test")

	data = np.array(res["test"]["disp"]).reshape(-1, 1)
	df_train = DataFrame(data, index=res["iter"], columns=["Test Disp."])
	m2nd_ax = df_train.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel("Statistical parity")

	fig.tight_layout()


if __name__ == "__main__":
	model = PreProcess("adult", "race")
	res=model.fit_transform()
	# draw(res)
