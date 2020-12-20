import torch
import numpy as np
from math import exp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
	import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from metric import Metric

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

def getdata(data,attr):
	data_funcs={
		'adult': load_preproc_data_adult,
		'german': load_preproc_data_german,
		'compas': load_preproc_data_compas
	}
	metadata=data_funcs[data]()

	X=metadata.features
	s=X[:,metadata.feature_names.index(attr)].reshape(-1)
	y=metadata.labels.reshape(-1)
	X=np.delete(X,metadata.feature_names.index(attr),axis=1)
	X=np.hstack([s.reshape(-1,1),X])
	return X,y.reshape(-1,1)

def lrc(data='adult',attr='sex'):

	X,y=getdata(data,attr)
	print(X.shape)

	X=torch.tensor(X).float()
	y=torch.tensor(y).float()

	model = LogisticRegression(input_dim=X.shape[1])
	optim = torch.optim.Adam(model.parameters(), lr=1E-4)
	loss_func=torch.nn.BCELoss()

	indices=list(range(0,X.shape[0]))
	
	n_epoch=10000
	batch_size=128
	for epoch in range(0, n_epoch):

		# if batch_size is not None:
		# 	idx=np.random.choice(indices, size=batch_size, replace=False).tolist()
		# 	batch_X=X[idx,:]
		# 	batch_y=y[idx,:]
		# else:	

		optim.zero_grad()
		y_pred=model(X)
		loss=loss_func(y_pred, y)
		loss.backward()
		optim.step()

		if epoch%10==0:
			metric=Metric(pred=y_pred.tolist(), true=y.tolist())
			print('Acc. =', metric.accuracy())
			print(model.linear_model.weight.grad)

def loss_fair(X,w):
	u_down=(1.0-X[:,0]).sum()
	v_down=X[:,0].sum()
	u_up=torch.tensor(0.0)
	v_up=torch.tensor(0.0)
	for i in range(0,X.shape[0]):
		u_up=u_up+torch.sigmoid(torch.matmul(X[i],w))*(1.0-X[i,0])
		v_up+=torch.sigmoid(torch.matmul(X[i],w))*X[i,0]
	loss=torch.square((u_up/u_down)-(v_up/v_down))
	return loss

def loss_robust_offset(X,y,w):
	loss=torch.tensor(0.0)
	for i in range(0, X.shape[0]):
		gradx=(y[i][0]-torch.sigmoid(torch.matmul(X[i],w)))*w
		loss+=torch.sum(torch.square(gradx))
	return loss/X.shape[0]

def loss_robust_switch(X,y,w,eps=0.1):
	loss=torch.tensor(0.0)
	for i in range(0, X.shape[0]):
		gradx=eps*(y[i][0]-torch.sigmoid(torch.matmul(X[i],w)))*w
		loss+=torch.sigmoid(torch.matmul(X[i]+gradx,w)*torch.matmul(X[i],w))
	return -loss/X.shape[0]


def main(data='adult',attr='sex'):

	X,y=getdata(data,attr)
	X=torch.tensor(X, requires_grad=False).float()
	y=torch.tensor(y, requires_grad=False).float()

	for i in range(0,5):
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
		print(angle, end=' ')

		w.grad=None
		lossr=loss_robust_switch(X,y,w,eps=0.0)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		print(angle, end=' ')

		w.grad=None
		lossr=loss_robust_switch(X,y,w,eps=0.1)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		print(angle, end=' ')

		w.grad=None
		lossr=loss_robust_switch(X,y,w,eps=0.2)
		lossr.backward()
		gradr=np.array(w.grad.tolist())
		angle=get_angle(gradf,gradr)
		angles.append(angle)
		grads.append(gradr.tolist())
		print(angle, end=' ')

		print()

		f=open('./result/gradient/angles_%s_%s.txt'%(data,attr),'a')
		f.write(str(angles)+'\n')
		f.close()

		f=open('./result/gradient/grads_%s_%s.txt'%(data,attr),'a')
		f.write(str(grads)+'\n')
		f.close()


if __name__=='__main__':
	main('adult','sex')







