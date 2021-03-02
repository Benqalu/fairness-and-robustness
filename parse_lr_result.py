import os
import torch
import numpy as np
import pandas as pd
from utils import get_data
from FaroLR import loss_fairness, loss_robustness
from matplotlib import pyplot as plt

def grad_robustness(X,y,w):
	w=torch.tensor(w,dtype=torch.float32,requires_grad=True)
	loss=loss_robustness(X,y,w)
	loss.backward()
	return np.array(w.grad.reshape(-1).tolist())

def grad_fairness(X,y,w):
	w=torch.tensor(w,dtype=torch.float32,requires_grad=True)
	loss=loss_fairness(X,y,w,tp=True)
	loss.backward()
	return np.array(w.grad.reshape(-1).tolist())

def angle(X,y,w):
	v1=grad_fairness(X,y,w)
	v2=grad_robustness(X,y,w)
	if np.sum(np.abs(v1))>0.0:
		u1=v1/np.linalg.norm(v1)
	else:
		u1=v1
	if np.sum(np.abs(v2))>0.0:
		u2=v2/np.linalg.norm(v2)
	else:
		u2=v2
	dot=np.dot(u1,u2)
	angle=np.arccos(dot)
	degree=(angle/np.pi)*180
	return degree

def origin_accuracy(include_s=True, metric='acc'):

	path='./result/lr_positive_disp/'

	ret={}

	alphas=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
	betas=[0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

	fnames=os.listdir(path)
	fnames.sort()

	current_data=None
	current_attr=None

	suffix='sY' if include_s else 'sN'

	for fname in fnames:
		if 'report' not in fname:
			continue
		if suffix not in fname:
			continue
		print(fname)
		components=fname.split('_')
		data=components[1]
		attr=components[2]
		alpha=int(components[4][1:])/100
		beta=int(components[5][1:])/100

		if (current_data,current_attr)!=(data,attr):
			current_data=data
			current_attr=attr
			X,y=get_data(data,attr)
			X=np.hstack([X,np.ones(X.shape[0]).reshape(-1,1)])
			X=torch.tensor(X,dtype=torch.float32)
			y=torch.tensor(y,dtype=torch.float32)

		if data not in ret:
			ret[data]={}
		if attr not in ret[data]:
			ret[data][attr]=np.zeros((len(alphas),len(betas)))

		f=open(path+fname)
		report=eval(f.readline())
		f.close()

		if metric=='acc':
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['accuracy'][-1]
		elif metric=='disp':
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['disparity'][-1]
		elif metric=='attk':
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['attack']['attack']
		elif metric=='rad':
			epochs=[]
			angles=[]
			cumulated_loss=[]
			cumulated_sum=0.0
			last_cumulated_sum=None
			for i in range(0,300):
				loss = report['loss_utility'][i] + alpha*report['loss_fairness'][i] + beta*report['loss_robustness'][i]
				if len(cumulated_loss)>=10:
					cumulated_sum = cumulated_sum - cumulated_loss[0] + loss
					cumulated_loss.append(loss)
					del cumulated_loss[0]
				else:
					cumulated_sum += loss
					cumulated_loss.append(loss)
				if len(cumulated_loss)>=10 and last_cumulated_sum is not None:
					if abs(cumulated_sum-last_cumulated_sum)<1E-3:
						print(i)
						break
				if len(cumulated_loss)>=10:
					last_cumulated_sum=cumulated_sum
				epochs.append(i*10)
				angles.append(report['angle'][i])
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=np.mean(angles)
		elif metric=='w2':
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=np.sum(np.square(report['weight'][-1]))
		elif metric=='ul':
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['loss_utility'][-1]
		elif metric=='w2ul':
			ret[data][attr][alphas.index(alpha)][betas.index(beta)]=report['loss_utility'][-1]*np.sum(np.square(report['weight'][-1]))



	for data in ['adult', 'compas']:
		for attr in ['race', 'sex']:
			matrix=ret[data][attr]
			print(f'{data}_{attr}')
			for i in range(matrix.shape[0]):
				for j in range(matrix.shape[1]):
					if j==matrix.shape[1]-1:
						print('%0.4f'%matrix[i][j])
					else:
						print('%0.4f'%matrix[i][j],end='\t')
			print()

	return ret

if __name__=='__main__':
	origin_accuracy(include_s=False, metric='w2ul')