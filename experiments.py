import torch
import numpy as np
from time import time, sleep
from utils import get_data
from FaroLR import FaroLR, loss_fairness, loss_robustness

from matplotlib import pyplot as plt

class Experiments(object):
	def __init__(self,data,attr,alpha=0.0,beta=0.0,n_epoch=3000,bias=True):
		X,y=get_data(data,attr)
		if bias:
			X=np.hstack([X,np.ones(X.shape[0]).reshape(-1,1)])
		self._bias=bias
		self._X=torch.tensor(X,dtype=torch.float32)
		self._y=torch.tensor(y,dtype=torch.float32)
		self._timestamp=int(time())
		self._alpha=alpha
		self._beta=beta
		self._n_epoch=n_epoch
		self._data=data
		self._attr=attr

	def calc_angle(self,v1,v2,rad=False):
		u1=v1/np.linalg.norm(v1)
		u2=v2/np.linalg.norm(v2)
		dot=np.dot(u1,u2)
		angle=np.arccos(dot)
		if rad:
			return angle
		else:
			degree=(angle/np.pi)*180
			return degree

	def grad_robustness(self,w):
		w=torch.tensor(w,dtype=torch.float32,requires_grad=True)
		loss=loss_robustness(self._X,self._y,w)
		loss.backward()
		return np.array(w.grad.reshape(-1).tolist())

	def grad_fairness(self,w):
		w=torch.tensor(w,dtype=torch.float32,requires_grad=True)
		loss=loss_fairness(self._X,w)
		loss.backward()
		return np.array(w.grad.reshape(-1).tolist())

	def exec(self,n_epoch=3000):
		X,y=get_data(self._data,self._attr)
		model = FaroLR(fairness=self._alpha, robustness=self._beta, lr=1E-3, n_epoch=n_epoch, bias=self._bias, report=['weight','accuracy','disparity'], seed=24)
		model.fit(X,y)
		model.eval_attack(X,y)

		attack_effectiveness=model._report['attack']
		print(attack_effectiveness)

		w=model._report['weight']

		f=open(f'./result/logistic_regression/report_{self._data}_{self._attr}_f{"%03d"%int(self._alpha*100)}_r{"%03d"%int(self._beta*100)}_ep{self._n_epoch}_{self._timestamp}.txt','w')
		f.write(str(model._report))
		f.close()

		epoch=[]
		angle=[]

		for i in range(0,len(w)):
			epoch.append(i*10+1)
			v1=self.grad_fairness(w[i])
			v2=self.grad_robustness(w[i])
			angle.append(self.calc_angle(v1,v2))

		plt.clf()
		plt.xlabel('n_epoch')
		plt.ylabel('Angle (rad)')
		plt.title(f'{self._data}_{self._attr}')
		plt.plot(epoch,angle)
		plt.savefig(f'./result/logistic_regression/angle_{self._data}_{self._attr}_{self._timestamp}.pdf')
		# plt.show()

if __name__=='__main__':
	for data in ['adult', 'compas', 'hospital']:
		for attr in ['sex', 'race']:
			for alpha in [0.0, 0.1, 0.2, 0.3]:
				for beta in [0.0, 0.01, 0.02, 0.03]:
					exp=Experiments(data,attr,alpha=alpha,beta=beta,n_epoch=3000,bias=True)
					exp.exec()
					del exp
					sleep(10)
