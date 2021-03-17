import time
import torch
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from art.classifiers.scikitlearn import ScikitlearnLogisticRegression as ART_LR
from art.attacks.evasion import ProjectedGradientDescent as ProjectedGradientDescentAttack

from metric import Metric

class TorchNNCore(torch.nn.Module):
	def __init__(self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU):
		super(TorchNNCore, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		struct = [inps]+hiddens+[1]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i], bias=bias))
			if i==len(struct)-1:
				layers.append(torch.nn.Sigmoid())
			else:
				layers.append(hidden_activation())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		output = torch.hstack([1-output,output])
		return output

def load_split(data,attr):
	df=pd.read_csv(f'./data/{data}_train.csv')
	columns=list(df.columns)
	data_train=df.to_numpy()
	s_train=data_train[:,columns.index(attr)]
	y_train=data_train[:,-1]
	X_train=np.delete(data_train[:,:-1],columns.index(attr),axis=1)

	df=pd.read_csv(f'./data/{data}_test.csv')
	columns=list(df.columns)
	data_test=df.to_numpy()
	s_test=data_test[:,columns.index(attr)]
	y_test=data_test[:,-1]
	X_test=np.delete(data_test[:,:-1],columns.index(attr),axis=1)

	return {'X':X_train,'s':s_train,'y':y_train}, {'X':X_test,'s':s_test,'y':y_test}

def encode(z):
	return np.eye(2)[z.astype(int)]

def main(data, attr, method='FGSM', eps=0.1, defense=None, seed=None):
	
	train, test = load_split(data, attr)

	clf=LogisticRegression(max_iter=500)
	art=ART_LR(clf)
	art.fit(train['X'],encode(train['y']))

	train['y_pred']=art.predict(train['X'])[:,1].round()
	metric=Metric(true=train['y'], pred=train['y_pred'])
	print('Train:',metric.accuracy())

	test['y_pred']=art.predict(test['X'])[:,1].round()
	metric=Metric(true=test['y'], pred=test['y_pred'])
	print('Train:',metric.accuracy())

	attack=ProjectedGradientDescentAttack(
		estimator=art,
		eps=eps,
		eps_step=eps*0.1,
		max_iter=10,
		verbose=False
	)

	train['X_adv']=attack.generate(x=train['X'],y=train['y'])
	train['y_adv_pred']=art.predict(train['X_adv'])[:,1].round()
	metric=Metric(true=train['y'], pred=train['y_adv_pred'])
	print('Train Adv.:',metric.accuracy())

	test['X_adv']=attack.generate(x=test['X'])#,y=test['y'])
	test['y_adv_pred']=art.predict(test['X_adv'])[:,1].round()
	metric=Metric(true=test['y'], pred=test['y_adv_pred'])
	print('Test Adv.:',metric.accuracy())

	if defense is not None:
		n_advs = defense if type(defense) is int else int(train['X'].shape[0]*defense)
		idx=np.random.choice(np.arange(0,train['X'].shape[0]),n_advs,replace=False)
		X_adv=np.vstack([train['X'],train['X_adv'][idx,:]])
		y_adv=np.hstack([train['y'],train['y'][idx]])
		s_adv=np.hstack([train['s'],train['s'][idx]])
		art.fit(X_adv,encode(y_adv))

	train['y_def_pred']=art.predict(train['X_adv'])[:,1].round()
	metric=Metric(true=train['y'], pred=train['y_def_pred'])
	print('Train Def.:',metric.accuracy())

	test['y_def_pred']=art.predict(test['X_adv'])[:,1].round()
	metric=Metric(true=test['y'], pred=test['y_def_pred'])
	print('Test Def.:',metric.accuracy())


if __name__=='__main__':
	seed=int(time.time())
	print('Seed:',seed)
	data='adult'
	attr='race'

	main(data,attr,method='FGSM',defense=1.0,seed=seed,eps=0.1)
