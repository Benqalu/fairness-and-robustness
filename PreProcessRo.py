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
			'X':torch.tensor(self._train_np['X'], dtype=torch.float),
			'y':torch.tensor(self._train_np['y'].reshape(-1,1), dtype=torch.float, requires_grad=True),
			'y_':torch.tensor(self._train_np['y'].reshape(-1,1), dtype=torch.float),
			'c':torch.tensor(self._train_np['c'].reshape(-1,1), dtype=torch.float),
		}
		
		self._test = {
			'X':torch.tensor(self._test_np['X'], dtype=torch.float),
			'y':torch.tensor(self._test_np['y'].reshape(-1,1), dtype=torch.float),
		}


	def _BCELoss(self, y_pred, y, reduction=True):
		if reduction:			
			return -torch.mean(
				y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred)
			)
		else:
			return -(y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred))

	def _DISPLoss(self, c, y_pred):
		return torch.square(torch.sum(c * y_pred))

	def _AccDisp(self, y, y_pred, s):
		metric = Metric(true=y.reshape(-1).tolist(), pred=y_pred.reshape(-1).tolist())
		return metric.accuracy(), metric.positive_disparity(s=s)

	def _Scaler(self, a_):
		a=np.array(a_)
		a_min = a.min()
		a_max = a.max()
		if a_min == a_max:
			return a
		else:
			return (a - a_min)/(a_max - a_min)*2 - 1

	def AdvExp(self, X, y=None, method="FGSM", eps=0.1):
		if type(X) is torch.Tensor:
			X_ = X.clone().detach().requires_grad_(True)
		else:
			X_ = torch.tensor(X, dtype=torch.float, requires_grad=True)
		if method == "FGSM":
			y_pred = self._model(X_)
			if y is not None:
				y_ = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
			else:
				y_ = torch.round(y_pred).detach()
			loss = self._BCELoss(y_pred, y_)
			noise = eps * torch.sign(torch.autograd.grad(loss, X_)[0])
			return (X_ + noise).detach()
		elif method == "PGD":
			if y is not None:
				y_ = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
			else:
				y_ = None
			for i in range(0, 10):
				y_pred = self._model(X_)
				if y_ is None:
					y_ = torch.round(y_pred).detach()
				loss = self._BCELoss(y_pred, y_)
				noise = (eps * 0.1) * torch.sign(torch.autograd.grad(loss, X_)[0])
				X_ = (X_ + noise).detach().requires_grad_(True)
			return X_.detach()

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
			hidden_activation=torch.nn.ReLU,
		)
		self._model = model

		optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
		loss_func = torch.nn.BCELoss()

		self._train['X_adv'] = None
		self._train['y_adv'] = None

		chosen = set()

		for it in range(0, self._max_iter):

			if len(chosen) == self._train['X'].shape[0]:
				break

			if self._train['X_adv'] is not None:
				X_train = torch.vstack([self._train['X'], self._train['X_adv']])
				y_train = torch.vstack([self._train['y'], self._train['y_adv']])
			else:
				X_train = self._train['X']
				y_train = self._train['y']

			# BEGIN: Train modified model
			tolerence = 10
			last_loss = None
			for epoch in range(0, 1000):
				optim.zero_grad()
				y_pred = model(X_train)
				loss = self._BCELoss(y_pred, y_train)
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
		
			# BEGIN: Pre-processing of robustness
			X_adv = self.AdvExp(X=self._train['X'], y=self._train['y'], method=self._method)
			y_adv_pred = model(X_adv)
			loss = self._BCELoss(y_adv_pred, self._train['y'], reduction=False).detach().numpy().reshape(-1)
			indices = np.argsort(loss)
			actual_indices = []
			for item in indices:
				if item not in chosen:
					chosen.add(item)
					actual_indices.append(item)
					if len(actual_indices) == self._k:
						break
			if self._train['X_adv'] is None:
				self._train['X_adv'] = X_adv[actual_indices,:].clone().detach()
				self._train['y_adv'] = self._train['y'][actual_indices,:].clone().detach()
			else:
				self._train['X_adv'] = torch.vstack([
					self._train['X_adv'],
					X_adv[actual_indices,:].clone().detach()
				])
				self._train['y_adv'] = torch.vstack([
					self._train['y_adv'],
					self._train['y'][actual_indices,:].clone().detach()
				])
			# END: Pre-processing of robustness

			y_pred = model(self._train['X'])
			metric = Metric(true=self._train_np['y'], pred=y_pred.detach().numpy().reshape(-1))
			acc_train_org = metric.accuracy()
			disp_train_org = metric.positive_disparity(s=self._train_np['s'])
			y_pred_adv = model(self.AdvExp(X=self._train['X'],y=self._train['y'],method=self._method))
			acc_train_atk = Metric(true=self._train_np['y'], pred=y_pred_adv.detach().numpy().reshape(-1)).accuracy()
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

				X_test_adv = self.AdvExp(X=self._test['X'], y=None, method=self._method)
				y_test_pred_atk = model(X_test_adv)
				acc_test_atk = Metric(
					true=self._test_np['y'], pred=y_test_pred_atk.detach().numpy().reshape(-1)
				).accuracy()

				print(
					"Test: (%.4f, %.4f, %.4f), Rc:%.2f"
					% (acc_test_org, acc_test_atk, disp_test_org, self._train['X_adv'].shape[0] / self._train['X'].shape[0])
				)
				res["test"]["orig"].append(acc_test_org)
				res["test"]["attk"].append(acc_test_atk)
				res["test"]["disp"].append(disp_test_org)
			# END: Testing
			
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
		data = 'adult'
		attr = 'race'
		method = 'FGSM'
		dF = 1.00
		dR = 0.4
		k = 0.0015

	import time
	seed = int(time.time())

	print((data, attr, method, dF, dR))
	print('Seed is %d'%seed)

	model = PreProcFlip(data, attr, k=k, max_iter=1000, seed=seed, dR=dR, dF=dF)
	res = model.fit_transform()

	res['data'] = data
	res['attr'] = attr
	res['method'] = method
	res['dF'] = dF
	res['dR'] = dR
	res['k'] = k
	res['seed'] = seed

	import json

	f=open(f'./result/preproc/Robustness_{data}_{attr}.txt','a')
	f.write(json.dumps(res)+'\n')
	f.close()
