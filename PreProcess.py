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
	def __init__(self, data, attr, method='FGSM', k=10, seed=None, max_iter=1000, dR=0.1, dF=0.1):
		self._data = data
		self._attr = attr
		self._seed = seed
		self._max_iter = max_iter
		self._epsilon = 0.1
		self._dR = dR
		self._dF = dF
		self._method = method
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

		self._train_np, self._test_np = load_split(data, attr)
		self._train_np['y_'] = self._train_np['y'].copy()

		if type(k) is int:
			self._k = k
		elif type(k) is float:
			self._k = np.ceil(self._train_np['X'].shape[0] * k).astype(int)
		else:
			raise ValueError('k must be float or int')

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

	def _Scaler(self, a):
		a_min = a.min()
		a_max = a.max()
		if a_min == a_max:
			return a
		else:
			return (a - a_min)/(a_max - a_min)

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
			hiddens=[],
			seed=self._seed,
			hidden_activation=torch.nn.ReLU,
		)

		optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
		loss_func = torch.nn.BCELoss()

		chosen = set()

		for it in range(0, self._max_iter):

			# BEGIN: Train modified model
			tolerence = 10
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
			y_grad = gradient(self._BCELoss(y_pred, self._train['y']), self._train['y'])[0].detach().numpy().reshape(-1)
			influence_to_utility = -y_grad * np.sign(self._train_np['y'] - 0.5)
			metric = Metric(true=self._train_np['y'], pred=y_pred.detach().numpy().reshape(-1))
			acc_train_org = metric.accuracy()
			disp_train_org = metric.positive_disparity(s=self._train_np['s'])

			ranking_utility = np.zeros(self._train['y'].shape[0])
			ranking_utility[np.argsort(influence_to_utility)] = np.arange(0, self._train['y'].shape[0]) + 1
			influence_to_utility = self._Scaler(influence_to_utility)
			# END: Influences of utility

			# BEGIN: Influences? of fairness
			metric = Metric(pred=y_pred.detach().numpy(), true=self._train_np['y_'])
			disp = metric.positive_disparity(s=self._train_np['s'], absolute=False)
			influence_to_fairness = np.sign(
				disp * (self._train_np['s'] - 0.5) * (self._train_np['y_'] - 0.5)
			)
			# ) * (0.5 - abs(y_pred_np - 0.5))
			if influence_to_fairness.min() >= 0:
				break

			ranking_fairness = np.zeros(self._train['y'].shape[0])
			ranking_fairness[np.argsort(influence_to_fairness)] = np.arange(0, self._train['y'].shape[0]) + 1
			influence_to_fairness = self._Scaler(influence_to_fairness)
			# END: Influences? of fairness
		
			
			# BEGIN: Influences of robustness
			if self._method == 'FGSM':
				self._train['X'].grad = None
				self._train['y'].grad = None
				y_pred = model(self._train['X'])
				noise = torch.sign(gradient(self._BCELoss(y_pred, self._train['y']), self._train['X'])[0])
				y_pred_atk = model(self._train['X'] + 0.1 * noise)
				y_grad = gradient(self._BCELoss(y_pred_atk, self._train['y']), self._train['y'])[0].detach().numpy().reshape(-1)
				influence_to_robustness = - y_grad * np.sign(self._train_np['y']-0.5)
				acc_train_atk = Metric(true=self._train_np['y'], pred=y_pred_atk.detach().numpy().reshape(-1)).accuracy()
				if influence_to_robustness.min() >= 0:
					break

				ranking_robustness = np.zeros(self._train['y'].shape[0])
				ranking_robustness[np.argsort(influence_to_robustness)] = np.arange(0, self._train['y'].shape[0]) + 1
				influence_to_robustness = self._Scaler(influence_to_robustness)
			elif self._method == 'PGD':
				pass
			else:
				raise RuntimeError
			# END: Influences of robustness

			print(
				"Iter: %d, Train: (%.4f, %.4f, %.4f)"
				% (it + 1, acc_train_org, acc_train_atk, disp_train_org),
				end=", ",
			)
			res["train"]["orig"].append(acc_train_org)
			res["train"]["attk"].append(acc_train_atk)
			res["train"]["disp"].append(disp_train_org)

			Mode = None
			if (1.0 - acc_train_atk) > self._dR and acc_train_atk < acc_train_org - 0.005:
				influence_score = influence_to_robustness
				Mode = 'Robustness'
			elif disp_train_org > self._dF:
				influence_score = influence_to_fairness
				Mode = 'Fairness'
			else:
				Mode = None

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
					"Test: (%.4f, %.4f, %.4f), Mode: %s"
					% (acc_test_org, acc_test_atk, disp_test_org, Mode)
				)
				res["test"]["orig"].append(acc_test_org)
				res["test"]["attk"].append(acc_test_atk)
				res["test"]["disp"].append(disp_test_org)
			# END: Testing

			

			# BEGIN: Choosing Strategy
			if Mode is None:
				break

			influence = np.column_stack(
				(
					# influence_to_fairness,
					# influence_to_robustness,
					# influence_to_utility,
					influence_score,
					list(range(0, self._train['y'].shape[0])),
				)
			)
			influence = influence.tolist()
			influence.sort()
			# END: Choosing Strategy

			# BEGIN: Flipping
			rest = self._k
			for item in influence:
				i = int(item[1])
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

	import sys
	if len(sys.argv)>=2:
		data = sys.argv[1]
		attr = sys.argv[2]
		method = sys.argv[3]
		dF = round(float(sys.argv[4]), 2)
		dR = round(float(sys.argv[5]), 2)
		k = 0.0015
	else:
		data = 'compas'
		attr = 'race'
		method = 'FGSM'
		dF = 0.03
		dR = 0.4
		k = 0.0015

	import time
	seed = int(time.time())

	print((data, attr, method, dF, dR))
	print('Seed is %d'%seed)

	model = PreProcFlip(data, attr, k=k, max_iter=500, seed=seed, dR=dR, dF=dF)
	res = model.fit_transform()

	res['data'] = data
	res['attr'] = attr
	res['method'] = method
	res['dF'] = dF
	res['dR'] = dR
	res['k'] = k
	res['seed'] = seed

	import json

	f=open(f'./result/preproc/FnR_Pre.txt','a')
	f.write(json.dumps(res)+'\n')
	f.close()
