import torch
import numpy as np
from time import time
from utils import get_data
from FaroLR import FaroLR, loss_fairness, loss_robustness

from matplotlib import pyplot as plt

class AngleTest(object):
	def __init__(self,data,attr,bias=True):
		X,y=get_data(data,attr)
		if bias:
			X=np.hstack([X,np.ones(X.shape[0]).reshape(-1,1)])
		self._bias=bias
		self._X=torch.tensor(X,dtype=torch.float32)
		self._y=torch.tensor(y,dtype=torch.float32)

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
		X,y=get_data('adult','sex')
		model = FaroLR(fairness=0.3, robustness=0.01, lr=1E-3, n_epoch=n_epoch, bias=self._bias, report=['weight'])
		model.fit(X,y)
		w=model._report['weight']

		epoch=[]
		angle=[]

		for i in range(0,n_epoch):
			epoch.append(i+1)
			v1=self.grad_fairness(w[i])
			v2=self.grad_robustness(w[i])
			angle.append(self.calc_angle(v1,v2))

		plt.plot(epoch,angle)
		plt.savefig(str(time())+'.pdf')
		plt.show()

if __name__=='__main__':
	test=AngleTest('adult','sex',bias=True)
	test.exec(n_epoch=3000)