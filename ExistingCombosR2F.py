import time
import numpy as np
import pandas as pd

from TorchAdversarial import TorchAdversarial

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

def main(data, attr, method='FGSM', eps=0.1, wR=0.1, seed=None):
	report={}

	train, test = load_split(data, attr)

	model = TorchAdversarial(lr=0.01, n_epoch=500, method=method, hiddens=[128], seed=seed)
	model.fit(train['X'], train['y'], train['s'], wR=wR)

	report['train'] = model.metrics(X=train['X'],y=train['y'],s=train['s'])
	report['test'] = model.metrics(X=test['X'],y=test['y'],s=test['s'])
	if method=='None':
		report['test_adv'] = None
	else:
		report['test_adv'] = model.metrics_attack(X=test['X'],y=test['y'],s=test['s'],method=method,use_y=False)

	return report
	

if __name__=='__main__':

	import json,sys,time

	if len(sys.argv)>1:
		data=sys.argv[1]
		attr=sys.argv[2]
		method=sys.argv[3]
		pidx=int(sys.argv[4])
	else:
		data='adult'
		attr='sex'
		method='FGSM'
		pidx=0

	wRs = {
		'FGSM': [0.1, 0.01, 0.05, 0.5, 1.0],
		'PGD': [0.1, 0.01, 0.05, 0.5, 1.0],
		'None': [0.1],
	}
	wR = wRs[method][pidx]

	seed=int(time.time())
	print('Seed:',seed)
	print(data, attr, method, wR)

	report=main(data,attr,method=method,seed=seed,wR=wR)
	report['seed']=seed
	report['data']=data
	report['attr']=attr
	report['method']=method
	report['wR']=wR
	report['pidx']=pidx

	# print(report)

	f=open('./result/existings/R2F.txt','a')
	f.write(json.dumps(report)+'\n')
	f.close()



	



