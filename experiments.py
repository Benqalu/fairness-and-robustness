import torch
import json
import numpy as np
from time import time, sleep
from utils import get_data
from FaroLR import FaroLR, loss_fairness, loss_robustness
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


class Experiments(object):
	def __init__(
		self,
		data,
		attr,
		prefix="inproc_positive_disp",
		wF=0.0,
		wR=0.0,
		n_epoch=1000,
		bias=True,
		recording=True,
		include_s=False,
		seed=24,
	):
		X, y = get_data(data, attr)
		self._bias = bias
		self._X = torch.tensor(X[:, 1:], dtype=torch.float32)
		if include_s:
			self._X_actual = torch.tensor(X, dtype=torch.float32)
		else:
			self._X_actual = torch.tensor(X[:, 1:], dtype=torch.float32)
		self._s = torch.tensor(X[:, 0], requires_grad=False)
		self._y = torch.tensor(y, dtype=torch.float32)
		self._timestamp = int(time())
		self._wF = wF
		self._wR = wR
		self._n_epoch = n_epoch
		self._data = data
		self._attr = attr
		self._prefix = prefix
		self._recording = recording
		self._include_s = include_s
		self._seed=seed
		np.random.seed(self._seed)
		torch.manual_seed(self._seed)

	def calc_angle(self, v1, v2, rad=False):
		u1 = v1 / np.linalg.norm(v1)
		u2 = v2 / np.linalg.norm(v2)
		dot = np.dot(u1, u2)
		angle = np.arccos(dot)
		if rad:
			return angle
		else:
			degree = (angle / np.pi) * 180
			return degree

	def grad_robustness(self, w):
		X = self._X_actual
		y = self._y
		w = torch.tensor(w, dtype=torch.float32, requires_grad=True)
		loss = loss_robustness(X, y, w)
		loss.backward()
		return np.array(w.grad.reshape(-1).tolist())

	def grad_fairness(self, w):
		X = self._X_actual
		s = self._s
		y = self._y
		w = torch.tensor(w, dtype=torch.float32, requires_grad=True)
		if "positive" in self._prefix:
			loss = loss_fairness(X, s, y, w, tp=False)
		elif "truepos" in self._prefix:
			loss = loss_fairness(X, s, y, w, tp=True)
		loss.backward()
		return np.array(w.grad.reshape(-1).tolist())

	def exec(self):
		np.random.seed(self._seed)
		torch.manual_seed(self._seed)

		print('include_s:',self._include_s)
		n_epoch = self._n_epoch
		X, y = get_data(self._data, self._attr)
		X, X_test, y, y_test=train_test_split(X,y,test_size=0.30)

		if "positive" in self._prefix:
			model = FaroLR(
				fairness=self._wF,
				robustness=self._wR,
				lr=0.1,
				n_epoch=n_epoch,
				bias=self._bias,
				report=[
					"weight",
					"accuracy",
					"disparity",
					"loss_utility",
					"loss_fairness",
					"loss_robustness",
				],
				tp_fairness=False,
				seed=24,
			)
		elif "truepos" in self._prefix:
			model = FaroLR(
				fairness=self._wF,
				robustness=self._wR,
				lr=0.1,
				n_epoch=n_epoch,
				bias=self._bias,
				report=[
					"weight",
					"accuracy",
					"disparity",
					"loss_utility",
					"loss_fairness",
					"loss_robustness",
				],
				tp_fairness=True,
				seed=24,
			)
		report = model.fit(X, y, X_test=X_test, y_test=y_test, include_s=self._include_s)
		print({'train':report['attack'], 'test':report['attack_test']})

		w = report["weight"]

		if self._recording:

			if self._include_s:
				suffix='sY'
			else:
				suffix='sN'

			epoch = []
			angle = []

			for i in range(0, len(w)):
				v1 = self.grad_fairness(w[i])
				v2 = self.grad_robustness(w[i])
				angle.append(self.calc_angle(v1, v2))

			# plt.clf()
			# plt.xlabel("n_epoch")
			# plt.ylabel("Angle (rad)")
			# plt.title(f"{self._data}_{self._attr}")
			# plt.plot(epoch, angle)
			# plt.savefig(
			# 	f'./result/{self._prefix}/angle_{self._data}_{self._attr}_{suffix}_f{"%03d"%int(self._wF*100)}_r{"%03d"%int(self._wR*100)}_ep{self._n_epoch}_{self._timestamp}.pdf'
			# )
			
			report['angle']=angle

			f = open(
				f'./result/{self._prefix}/report_{self._data}_{self._attr}_{suffix}_f{"%03d"%int(self._wF*100)}_r{"%03d"%int(self._wR*100)}_ep{self._n_epoch}_{self._timestamp}.txt',
				"w",
			)
			f.write(json.dumps(report))
			f.close()


if __name__ == "__main__":

	import sys
	argv=sys.argv

	testing=False

	if testing:

		exp = Experiments(
			'compas',
			'race',
			wF=0.0,
			wR=0.01,
			n_epoch=500,
			bias=True,
			prefix="inproc_positive_disp",
			recording=True,
			include_s=False
		)
		exp.exec()

	else:

		if len(argv)>1 and argv[1].lower().strip()=='y':
			include_s=True
		else:
			include_s=False

		print(argv)

		print('Include_s:',include_s)

		for data in ["adult", "compas"]:#, "hospital"]:
			for attr in ["sex", "race"]:
				for wF in [round(i*0.01,2) for i in range(1,51)]:
					for wR in [round(i*0.01,2) for i in range(1,31)]:

						start_time = time()

						print('>>>', (data, attr, wF, wR))
						exp = Experiments(
							data,
							attr,
							wF=wF,
							wR=wR,
							n_epoch=500,
							bias=True,
							prefix="inproc_positive_disp",
							recording=True,
							include_s=False
						)
						exp.exec()
						del exp

						end_time = time()

						print('>>> Time cost: %.2fs'%(end_time - start_time))
						sleep(3)
