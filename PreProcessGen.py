import gzip
import torch
import pickle
import numpy as np
np.set_printoptions(suppress=False)
from torch.autograd import grad as gradient
from pandas import DataFrame
from matplotlib import pyplot as plt

from TorchAdversarial import TorchNNCore
from TorchAdversarialTmp import TorchAdversarial
from utils import load_split
from metric import Metric
from copy import deepcopy


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

		if type(k) is int:
			self._k = k
		elif type(k) is float:
			self._k = np.ceil(self._train_np['X'].shape[0] * k).astype(int)
		else:
			raise ValueError('k must be float or int')

		print('Flipping %.4f%% (%d) records every epoch'%(self._k*100.0/self._train_np['X'].shape[0], self._k))

		self._train = {
			'X':torch.tensor(self._train_np['X'], dtype=torch.float),
			'y':torch.tensor(self._train_np['y'].reshape(-1,1), dtype=torch.float),
			's':self._train_np['s']
		}
		
		self._test = {
			'X':torch.tensor(self._test_np['X'], dtype=torch.float),
			'y':torch.tensor(self._test_np['y'].reshape(-1,1), dtype=torch.float),
			's':self._test_np['s']
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

	def fit_transform(self, saveto=None):

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
			"valid": True,
		}

		self._train['X_adv'] = None

		chosen_R = []
		chosen_F = []
		max_R_value = 0.0
		last_X_adv = None

		for it in range(0, self._max_iter):

			model = TorchNNCore(
				inps=self._train['X'].shape[1],
				hiddens=[128],
				seed=self._seed,
				hidden_activation=torch.nn.ReLU,
			)
			self._model = model

			optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
			loss_func = torch.nn.BCELoss()

			if self._train['X_adv'] is not None:
				X_train = torch.vstack([self._train['X'], self._train['X_adv']]).detach()
				y_train = torch.vstack([self._train['y'], self._train['y'][chosen_R]]).detach()
			else:
				X_train = self._train['X'].detach()
				y_train = self._train['y'].detach()

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

			actual_indices_F = []
			infl = -np.ones(self._train['s'].shape[0])
			if ((len(res["test"]["disp"])==0) or (len(res["test"]["disp"])>0 and res["test"]["disp"][-1] > self._dF)) and len(chosen_F) < self._train['X'].shape[0]:
				# BEGIN: Pre-processing of fairness
				y_pred_np = model(self._train['X']).detach().numpy().reshape(-1)
				metric = Metric(pred=y_pred_np, true=self._train['y'].detach().numpy().reshape(-1))
				disp = metric.positive_disparity(s=self._train['s'], absolute=False)
				infl = np.sign(
					disp * (self._train['s'] - 0.5) * (self._train_np['y'] - 0.5)
				) * (0.5 - abs(y_pred_np - 0.5))
				indices = np.argsort(infl)
				chosen_F_set = set(chosen_F)
				for item in indices:
					if infl[item] >= 0:
						break
					if item not in chosen_F_set:
						actual_indices_F.append(item)
						if len(actual_indices_F) == self._k:
							break
				# chosen_F.extend(actual_indices_F)
				for item in actual_indices_F:
					self._train['y'][item][0] = 1.0 - self._train['y'][item][0]
				# END: Pre-processing of fairness
		
			actual_indices_R = []
			if (len(res["test"]["attk"])==0 or res["test"]["attk"][-1] < self._dR) and len(chosen_R) < self._train['X'].shape[0]:
				# BEGIN: Pre-processing of robustness
				X_adv = self.AdvExp(X=self._train['X'], y=self._train['y'], method=self._method)
				y_adv_pred = model(X_adv)
				loss = self._BCELoss(y_adv_pred, self._train['y'], reduction=False).detach().numpy().reshape(-1)
				indices = np.argsort(loss)
				chosen_R_set = set(chosen_R)
				for item in indices:
					if item not in chosen_R_set:
						actual_indices_R.append(item)
						if len(actual_indices_R) == self._k:
							break
				# chosen_R.extend(actual_indices_R)
				if self._train['X_adv'] is None:
					self._train['X_adv'] = X_adv[actual_indices_R,:].detach()
				else:
					last_X_adv = self._train['X_adv'].detach()
					self._train['X_adv'] = torch.vstack([
						self._train['X_adv'],
						X_adv[actual_indices_R,:].clone().detach()
					]).detach()
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
			if 1==1:
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
					"Test: (%.4f, %.4f, %.4f), Coverage: (R:%.2f, F:%.2f)"
					% (acc_test_org, acc_test_atk, disp_test_org, 
						len(chosen_R) / self._train['X'].shape[0],
						len(chosen_F) / self._train['X'].shape[0],
					)
				)
				res["test"]["orig"].append(acc_test_org)
				res["test"]["attk"].append(acc_test_atk)
				res["test"]["disp"].append(disp_test_org)
				if res["test"]["attk"][-1] > max_R_value:
					max_R_value = res["test"]["attk"][-1]
			# END: Testing

			if res["test"]["attk"][-1] < max_R_value - 0.1:
				res["valid"]=False
				break
			if min(infl)>=0 and res["test"]["disp"][-1] > self._dF:
				res["valid"]=False
				break

			if self._dR >= res["test"]["orig"][0] + 0.15:
				res["valid"]=False
				break
			if res["test"]["disp"][0] <= self._dF - 0.05:
				res["valid"]=False
				break

			if res["test"]["attk"][-1] >= self._dR and res["test"]["disp"][-1] <= self._dF:
				break
			if len(actual_indices_F)==0 and len(actual_indices_R)==0:
				break

			chosen_F.extend(actual_indices_F)
			chosen_R.extend(actual_indices_R)

		if last_X_adv is not None:
			self._downstream_train ={
				'X': torch.vstack([self._train['X'], last_X_adv]).detach(),
				'y': torch.vstack([self._train['y'], self._train['y'][chosen_R]]).detach(),
				's': np.hstack([self._train['s'], self._train['s'][chosen_R]]),
				'idx': chosen_R
			}
		else:
			self._downstream_train ={
				'X': self._train['X'].detach(),
				'y': self._train['y'].detach(),
				's': self._train['s'],
				'idx': chosen_R
			}

		self._valid = res['valid']

		save_res = deepcopy(res)

		if saveto is not None and res['valid']:
			save_res['seed'] = self._seed
			save_res['dR'] = self._dR
			save_res['dF'] =self._dF
			save_res['downstream_train'] = self._downstream_train
			save_res['downstream_test'] = self._test
			with open('./result/predata/'+saveto, 'wb') as handle:
				pickle.dump(save_res, handle)

		return res

	def downstreams(self, loadfrom = None):

		# if not self._valid:
		# 	return None

		if loadfrom is not None:
			with open('./result/predata/'+loadfrom, 'rb') as handle:
				res = pickle.load(handle)
				self._seed = res['seed']
				self._downstream_train = res['downstream_train']
				self._downstream_test = res['downstream_test']

		hidden_candidates = [[], [128], [128, 128], [256, 128]]
		res = []

		for hidden in hidden_candidates:

			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

			model = TorchNNCore(
				inps=self._downstream_train['X'].shape[1],
				hiddens=hidden,
				seed=self._seed,
				hidden_activation=torch.nn.ReLU,
			)
			self._model = model

			optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
			loss_func = torch.nn.BCELoss()

			tolerence = 10
			last_loss = None
			for epoch in range(0, 1000):
				optim.zero_grad()
				y_pred = model(self._downstream_train['X'])
				loss = self._BCELoss(y_pred, self._downstream_train['y'])
				this_loss = loss.tolist()
				if last_loss is not None:
					if this_loss > last_loss or abs(last_loss - this_loss) < 1e-5:
						tolerence -= 1
					if tolerence == 0:
						break
				last_loss = this_loss
				loss.backward()
				optim.step()

			y_test_pred = model(self._test['X'])
			metric = Metric(
				true=self._test['y'].reshape(-1).tolist(), pred=y_test_pred.detach().numpy().reshape(-1)
			)
			acc_test_org = metric.accuracy()
			disp_test_org = metric.positive_disparity(s=self._test['s'])

			X_test_adv = self.AdvExp(X=self._test['X'], y=None, method=self._method)
			y_test_pred_atk = model(X_test_adv)
			acc_test_atk = Metric(
				true=self._test['y'].reshape(-1).tolist(), pred=y_test_pred_atk.detach().numpy().reshape(-1)
			).accuracy()

			# print(acc_test_org, acc_test_atk, disp_test_org)
			res.append([acc_test_org, acc_test_atk, disp_test_org])

		return res


if __name__ == "__main__":

	import sys
	if len(sys.argv)>=2:
		data = sys.argv[1]
		attr = sys.argv[2]
		method = sys.argv[3]
		dF = round(float(sys.argv[4]), 3)
		dR = round(float(sys.argv[5]), 3)
		k = 0.003
	else:
		data = 'adult'
		attr = 'race'
		method = 'FGSM'
		dF = 0.06
		dR = 0.3
		k = 0.003

	import time, json
	seed = int(time.time())

	print((data, attr, method, dF, dR))
	print('Seed is %d'%seed)

	model = PreProcFlip(data, attr, k=k, method=method, max_iter=1000, seed=seed, dR=dR, dF=dF)
	res=model.fit_transform(saveto=f'{data}_{attr}_{method}_{seed}.pre')
	# model.downstreams(loadfrom='adult_race_FGSM_1620793926.pre') 


	# res['data'] = data
	# res['attr'] = attr
	# res['method'] = method
	# res['dF'] = dF
	# res['dR'] = dR
	# res['k'] = k
	# res['seed'] = seed

	# f=open(f'./result/preproc/FnR.txt','a')
	# f.write(json.dumps(res)+'\n')
	# f.close()
