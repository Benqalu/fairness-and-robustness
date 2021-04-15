import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gzip
import torch
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# from aif360.datasets import AdultDataset, CompasDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from ExistingApproaches.preprocessing import DisparateImpactRemover

from TorchAdversarial import TorchAdversarial
from metric import Metric

global_seed = None

def reset_seed(seed):
	if seed is not None:
		global_seed=seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		tf.random.set_random_seed(seed)

def load_data(data,attr):
	if data=='adult':
		with gzip.open(f'./data/{data}_train.pkl.gz', 'rb') as handle:
			data_train = pickle.load(handle)
		with gzip.open(f'./data/{data}_test.pkl.gz', 'rb') as handle:
			data_test = pickle.load(handle)
		if attr=="sex":
			privileged_groups = [{'sex': 1}]
			unprivileged_groups = [{'sex': 0}]
		elif attr=="race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
		else:
			raise RuntimeError('Unknown attr.')
	elif data=='compas':

		with gzip.open(f'./data/{data}_train.pkl.gz', 'rb') as handle:
			data_train = pickle.load(handle)
		with gzip.open(f'./data/{data}_test.pkl.gz', 'rb') as handle:
			data_test = pickle.load(handle)

		if attr == "sex":
			privileged_groups = [{'sex': 0}]
			unprivileged_groups = [{'sex': 1}]
		elif attr == "race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
		else:
			raise RuntimeError('Unknown attr.')
	else:
		raise RuntimeError('Unknown data.')

	return data_train, data_test, privileged_groups, unprivileged_groups

def get_Xsy(data,attr,del_s=True):
	X=data.features
	y=data.labels.reshape(-1)
	s_index=data.feature_names.index(attr)
	s=data.features[:,s_index].reshape(-1)
	if del_s:
		X=np.delete(X,s_index,axis=1)
	weight=data.instance_weights.reshape(-1)
	return X,s,y,weight

def fairness_reweighing(data, attr, method='FGSM', wF=0.0, wR=0.0):

	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)
	
	preproc = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
	data_train_proc = preproc.fit_transform(data_train)

	X,s,y,weight=get_Xsy(data_train_proc,attr,del_s=False)
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	model = TorchAdversarial(lr=0.01, n_epoch=500, method=method, hiddens=[128], seed=global_seed)
	if wF == 0.0 or wF is None:
		model.fit(X,y,s,wR=wR)
	else:
		model.fit(X,y,s,weight,wR=wR)
	report={
		'train':model.metrics(X=X,y=y,s=s),
		'test':model.metrics(X=X_test,y=y_test,s=s_test),
		'test_adv':model.metrics_attack(X=X_test,y=y_test,s=s_test,method=method,use_y=False),
	}
	# print('Training Acc.:', train_acc)
	# print('Testing Acc.:', test_acc)
	# print('Testing Atk Acc.:', test_adv_acc)
	return report

def fairness_disparate(data, attr, method='FGSM', mitigation=True, wF=1.0, wR=0.0):

	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)
	
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	if wF==0.0 or wF is None:
		X,s,y,weight=get_Xsy(data_train,attr,del_s=False)
	else:
		preproc = DisparateImpactRemover(repair_level=wF,sensitive_attribute=attr)
		data_train_proc = preproc.fit_transform(data_train)
		X,s,y,weight=get_Xsy(data_train_proc,attr,del_s=False)

	model = TorchAdversarial(lr=0.01, n_epoch=500, method=method, hiddens=[128], seed=global_seed)
	model.fit(X,y,s,weight,wR=wR)
	report={
		'train':model.metrics(X=X,y=y,s=s),
		'test':model.metrics(X=X_test,y=y_test,s=s_test),
		'test_adv':model.metrics_attack(X=X_test,y=y_test,s=s_test,method=method,use_y=False),
	}
	# print('Training Acc.:', train_acc)
	# print('Testing Acc.:', test_acc)
	# print('Testing Atk Acc.:', test_adv_acc)
	return report

if __name__=='__main__':

	import sys, json, time

	if len(sys.argv)>2:
		data = sys.argv[1].lower().strip()
		attr = sys.argv[2].lower().strip()
		method = sys.argv[3]
		func = locals()[sys.argv[4]]
		wF = round(float(sys.argv[5]),2)
		wR = round(float(sys.argv[6]),2)
		seed = int(time.time())
		print('Seed is %d.'%seed)
	else:
		data = 'compas'
		attr = 'race'
		method = 'FGSM'
		func = fairness_reweighing
		wF = 0.3
		wR = 0.1
		seed = None


	print(data, attr, method, func.__name__, 'wF=%.2f'%wF, 'wR=%.2f'%wR)

	reset_seed(seed)
	res=func(data,attr,method=method,wR=wR,wF=wF)

	report={
		'seed':seed,
		'data':data,
		'attr':attr,
		'method':method,
		'func':func.__name__,
		'result':res,
		'wR':wR,
		'wF':wF,
	}

	f=open('./result/existings/FnR.txt','a')
	f.write(json.dumps(report)+'\n')
	f.close()

