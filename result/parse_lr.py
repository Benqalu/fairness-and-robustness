import os
import torch
import numpy as np
import pandas as pd
from utils import get_data
from FaroLR import loss_fairness, loss_robustness

def grad_robustness(X,y,w):
	w=torch.tensor(w,dtype=torch.float32,requires_grad=True)
	loss=loss_robustness(X,y,w)
	loss.backward()
	return np.array(w.grad.reshape(-1).tolist())

def grad_fairness(X,y,w):
	w=torch.tensor(w,dtype=torch.float32,requires_grad=True)
	loss=loss_fairness(X,w)
	loss.backward()
	return np.array(w.grad.reshape(-1).tolist())

def angle(X,y,w):
	v1=grad_fairness(X,y,w)
	v2=grad_robustness(X,y,w)
	u1=v1/np.linalg.norm(v1)
	u2=v2/np.linalg.norm(v2)
	dot=np.dot(u1,u2)
	angle=np.arccos(dot)
	degree=(angle/np.pi)*180
	return degree

def origin_accuracy():

	ret={}

	alphas=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
	betas=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

	fnames=os.listdir('./logistic_regression/')
	fnames.sort()

	current_data=None
	current_attr=None
	for fname in fnames:
		if 'report' not in fname:
			continue
		components=fname.split('_')
		data=components[1]
		attr=components[2]
		alpha=int(components[3][1:])/100
		beta=int(components[4][1:])/100

		if (current_data,current_attr)!=(data,attr):
			current_data=data
			current_attr=attr
			X,y=get_data(data,attr)
			X=torch.tensor(X,dtype=torch.float32)
			y=torch.tensor(y,dtype=torch.float32)

		if data not in ret:
			ret[data]={}
		if attr not in ret[data]:
			ret[data][attr]=np.zeros((len(alphas),len(betas)))

		f=open('./logistic_regression/'+fname)
		report=eval(f.readline())
		f.close()

		# ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['accuracy'][-1]
		# ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['disparity'][-1]
		# ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['attack']['attack']
		ret[data][attr][alphas.index(alpha)][betas.index(beta)]=angle(X,y,report['weight'][-1])

	for data in ['adult', 'compas', 'hospital']:
		for attr in ['race', 'sex']:
			matrix=ret[data][attr]
			print(f'{data}_{attr}')
			for i in range(matrix.shape[0]):
				for j in range(matrix.shape[1]):
					if j==matrix.shape[1]-1:
						print('%.4f'%matrix[i][j])
					else:
						print('%.4f'%matrix[i][j],end='\t')
			print()

	return ret

if __name__=='__main__':
	origin_accuracy()