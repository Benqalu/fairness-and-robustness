import time
import numpy as np
import pandas as pd

from TorchAttackable import TorchNeuralNetworks, TorchNNCore

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

def main(data, attr, method='FGSM', eps=0.1, defense=None, seed=None):
	report={
		'train':{
			'acc':None,
			'disp':None,
		},
		'train_adv':{
			'acc':None,
			'disp':None,
		},
		'test':{
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
	model = TorchNeuralNetworks(lr=0.1,n_epoch=500,hiddens=[128],seed=seed)
	model.fit(train['X'], train['s'], train['y'])

	test['X_adv_fgsm']=model.AdvExp(X=test['X'], y=None, eps=eps, method='FGSM')
	test['X_adv_pgd']=model.AdvExp(X=test['X'], y=None, eps=eps, method='PGD')

	acc, disp = model.metrics(X=train['X'],y=train['y'],s=train['s'])
	report['train']['acc']=acc
	report['train']['disp']=disp

	if defense is not None and defense>0.0:
		advexps=model.AdvExp(train['X'], train['y'], eps=eps, method=method)
		n_advs = defense if type(defense) is int else int(train['X'].shape[0]*defense)
		idx=np.random.choice(np.arange(0,train['X'].shape[0]),n_advs,replace=False)
		# print('Added %d adversarial examples （%.2f%%).'%(len(idx), 100*len(idx)/advexps.shape[0]))
		X_adv=np.vstack([train['X'],advexps[idx,:]])
		y_adv=np.hstack([train['y'],train['y'][idx]])
		s_adv=np.hstack([train['s'],train['s'][idx]])
		model.fit(X_adv,s_adv,y_adv)
		acc, disp = model.metrics(X=train['X'],y=train['y'],s=train['s'])
		report['train_adv']['acc']=acc
		report['train_adv']['disp']=disp


	acc, disp = model.metrics(X=test['X'],y=test['y'],s=test['s'])
	report['test']['acc']=acc
	report['test']['disp']=disp

	acc, disp = model.metrics(X=test['X_adv_fgsm'],y=test['y'],s=test['s'])
	report['test_fgsm']['acc']=acc
	report['test_fgsm']['disp']=disp

	acc, disp = model.metrics(X=test['X_adv_pgd'],y=test['y'],s=test['s'])
	report['test_pgd']['acc']=acc
	report['test_pgd']['disp']=disp

	return report
	

if __name__=='__main__':

	import json,sys,time

	if len(sys.argv)>1:
		data=sys.argv[1]
		attr=sys.argv[2]
		method=sys.argv[3]
		defense=float(sys.argv[4])
	else:
		data='adult'
		attr='sex'
		method='FGSM'
		defense=1.0

	seed=int(time.time())
	print('Seed:',seed)
	print(data, attr, method, defense)

	report=main(data,attr,method=method,defense=defense,seed=seed,eps=0.1)
	report['seed']=seed
	report['data']=data
	report['attr']=attr
	report['method']=method
	report['defense']=defense

	# print(report)

	f=open('./result/existings/R2F.txt','a')
	f.write(json.dumps(report)+'\n')
	f.close()



	



