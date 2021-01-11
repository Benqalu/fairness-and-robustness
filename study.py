import torch
import tqdm
import numpy as np
from math import exp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
	import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from metric import Metric
from utils import getdata

class LogisticRegression(torch.nn.Module):
	def __init__(self, input_dim):
		super(LogisticRegression, self).__init__()
		self.linear_model = torch.nn.Linear(input_dim,1,bias=False)
		self.activation=torch.nn.Sigmoid()
	def forward(self,x):
		return self.activation(self.linear_model(x))

def sigmoid(x):
	return 1.0/(1+exp(-x))

def get_angle(v1,v2):
	u1=v1/np.linalg.norm(v1)
	u2=v2/np.linalg.norm(v2)
	dot=np.dot(u1,u2)
	angle=np.arccos(dot)
	degree=(angle/np.pi)*180
	return degree

def loss_fair(X,w):
	u_down=torch.sum(1.0-X[:,0])
	u_up=torch.sum(torch.sigmoid(torch.matmul(X,w))*(1.0-X[:,0]))
	v_down=torch.sum(X[:,0])
	v_up=torch.sum(torch.sigmoid(torch.matmul(X,w))*X[:,0])
	loss=torch.abs((u_up/u_down)-(v_up/v_down))
	return loss

def loss_robust_offset(X,y,w):
	gradx=(y.reshape(-1)-torch.sigmoid(torch.matmul(X,w))).reshape(-1,1)*w
	loss=torch.mean(torch.sum(torch.square(gradx),axis=1))
	return loss

def loss_robust_switch(X,y,w,eps=0.1):
	gradx=eps*(y.reshape(-1)-torch.sigmoid(torch.matmul(X,w))).reshape(-1,1)*w
	loss=torch.mean(1.0-torch.sigmoid(torch.matmul(X+gradx,w))*torch.matmul(X,w))
	return loss

def main(data='adult',attr='sex'):

	X,y=getdata(data,attr)
	X=torch.tensor(X, requires_grad=False).float()
	y=torch.tensor(y, requires_grad=False).float()

	all_angles=[]
	all_grads=[]

	for i in tqdm.tqdm(range(0,10000)):
		w=torch.tensor(np.random.uniform(-1,1,X.shape[1]).tolist(), requires_grad=True)

		angles=[]
		grads=[]

		w.grad=None
		lossf=loss_fair(X,w)
		lossf.backward()
		gradf=np.array(w.grad.tolist())

		w.grad=None
		lossr=loss_robust_offset(X,y,w)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		# print(angle, end=' ')

		w.grad=None
		lossr=loss_robust_switch(X,y,w,eps=0.0)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		# print(angle, end=' ')

		w.grad=None
		lossr=loss_robust_switch(X,y,w,eps=0.1)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		# print(angle, end=' ')

		w.grad=None
		lossr=loss_robust_switch(X,y,w,eps=0.2)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		# print(angle, end=' ')

		# print()s

		all_angles.append(angles)
		all_grads.append(grads)


	f=open('./result/gradient/angles_%s_%s.txt'%(data,attr),'w')
	for i in range(0, 10000):
		f.write(str(all_angles[i])+'\n')
	f.close()

	f=open('./result/gradient/grads_%s_%s.txt'%(data,attr),'w')
	for i in range(0, 10000):
		f.write(str(all_grads[i])+'\n')
	f.close()


if __name__=='__main__':
	import sys
	if len(sys.argv)>=3:
		data=sys.argv[1]
		attr=sys.argv[2]
	else:
		data='adult'
		attr='sex'

	for data in ['adult','compas','hospital']:
		for attr in ['sex','race']:
			print(data,attr)
			main(data,attr)







