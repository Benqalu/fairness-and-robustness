import numpy as np
import pandas as pd

from TorchAttackable import TorchNeuralNetworks

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

def main(data, attr, method='FGSM', eps=0.1, defense=None):
	report={
		'train':{
			'acc':None,
			'disp':None,
		},
		'train_adv':{
			'acc':None,
			'disp':None,
		},
		'test_orig':{
			'acc':None,
			'disp':None,
		},
		'test_fgsm':{
			'acc':None,
			'disp':None,
		},
		'test_pgd':{
			'acc':None,
			'disp':None,
		}
	}

	train, test = load_split(data, attr)
	model = TorchNeuralNetworks(hiddens=[128])
	model.fit(train['X'], train['s'], train['y'])

	acc, disp = model.metrics(X=train['X'],y=train['y'],s=train['s'])
	report['train']['acc']=acc
	report['train']['disp']=disp

	if defense is not None and defense>0.0:
		advexps=model.advexp(train['X'], train['y'], eps=eps, method=method)
		print(advexps.shape)
		idx=np.random.choice(np.arange(0,train['X'].shape[0]),int(train['X'].shape[0]*defense),replace=False)
		X_adv=np.vstack([train['X'],advexps[idx,:]])
		print(X_adv.shape)
		y_adv=np.hstack([train['y'],train['y'][idx]])
		s_adv=np.hstack([train['s'],train['s'][idx]])
		model.fit(X_adv,s_adv,y_adv)
		acc, disp = model.metrics(X=train['X'],y=train['y'],s=train['s'])
		report['train_adv']['acc']=acc
		report['train_adv']['disp']=disp

	acc, disp = model.metrics(X=test['X'],y=test['y'],s=test['s'])
	report['test_orig']['acc']=acc
	report['test_orig']['disp']=disp

	acc, disp = model.metrics_attack(X=test['X'],y=test['y'],s=test['s'],method='FGSM')
	report['test_fgsm']['acc']=acc
	report['test_fgsm']['disp']=disp

	acc, disp = model.metrics_attack(X=test['X'],y=test['y'],s=test['s'],method='PGD')
	report['test_pgd']['acc']=acc
	report['test_pgd']['disp']=disp

	return report
	

if __name__=='__main__':
	report=main('compas','sex',method='PGD',defense=0.5)
	print(report)