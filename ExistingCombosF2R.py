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
from ExistingApproaches.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.preprocessing import DisparateImpactRemover

from TorchAttackable import TorchNeuralNetworks, TorchNNCore
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
		with gzip.open(f'./data/{data}_train.pkl.gz', 'rb') as handle:
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
		with gzip.open(f'./data/{data}_train.pkl.gz', 'rb') as handle:
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

def fairness_reweighing(data, attr, method='FGSM', mitigation=True, param=None):

	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)
	
	preproc = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
	data_train_proc = preproc.fit_transform(data_train)

	X,s,y,weight=get_Xsy(data_train_proc,attr,del_s=False)
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	model = TorchNeuralNetworks(lr=0.1,n_epoch=500,hiddens=[128],seed=global_seed)
	if mitigation:
		model.fit(X,s,y,weight)
	else:
		model.fit(X,s,y)
	train_acc, train_disp = model.metrics(X=X,y=y,s=s)
	test_acc, test_disp = model.metrics(X=X_test,y=y_test,s=s_test)
	test_adv_acc, test_adv_disp = model.metrics_attack(X=X_test,y=y_test,s=s_test,method=method,use_y=False)
	report={
		'train_acc':train_acc,
		'train_disp':train_disp,
		'test_acc':test_acc,
		'test_disp':test_disp,
		'test_adv_acc':test_adv_acc,
		'test_adv_disp':test_adv_disp,
	}
	# print('Training Acc.:', train_acc)
	# print('Testing Acc.:', test_acc)
	# print('Testing Atk Acc.:', test_adv_acc)
	return report

def fairness_disparate(data, attr, method='FGSM', mitigation=True, param=1.0):

	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)
	
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	if mitigation:
		preproc = DisparateImpactRemover(repair_level=param,sensitive_attribute=attr)
		data_train_proc = preproc.fit_transform(data_train)
		X,s,y,weight=get_Xsy(data_train_proc,attr,del_s=False)
	else:
		X,s,y,weight=get_Xsy(data_train,attr,del_s=False)
	model = TorchNeuralNetworks(lr=0.1,n_epoch=500,hiddens=[128],seed=global_seed)
	model.fit(X,s,y,weight)
	train_acc, train_disp = model.metrics(X=X,y=y,s=s)
	test_acc, test_disp = model.metrics(X=X_test,y=y_test,s=s_test)
	test_adv_acc, test_adv_disp = model.metrics_attack(X=X_test,y=y_test,s=s_test,method=method,use_y=False)
	report={
		'train_acc':train_acc,
		'train_disp':train_disp,
		'test_acc':test_acc,
		'test_disp':test_disp,
		'test_adv_acc':test_adv_acc,
		'test_adv_disp':test_adv_disp,
	}
	# print('Training Acc.:', train_acc)
	# print('Testing Acc.:', test_acc)
	# print('Testing Atk Acc.:', test_adv_acc)
	return report

def fairness_adversarial(data, attr, method='FGSM', mitigation=True, param=0.1):
	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)

	X,s,y,_=get_Xsy(data_train, attr)
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	sess = tf.Session()
	inproc=AdversarialDebiasing(
		privileged_groups=privileged_groups,
		unprivileged_groups=unprivileged_groups,
		scope_name='debiased_classifier',
		classifier_num_hidden_units=128,
		num_epochs=500,
		lr=0.1,
		debias=mitigation,
		batch_size=data_train.features.shape[0],
		sess=sess,
		adversary_loss_weight=param
	)
	inproc.fit(data_train)

	data_train_pred = inproc.predict(data_train)
	metric = Metric(true=data_train.labels.reshape(-1), pred=data_train_pred.labels.reshape(-1))
	train_acc=metric.accuracy()
	train_disp=metric.positive_disparity(s=s)
	# print('Training Acc.:', train_acc)

	data_test_pred = inproc.predict(data_test)
	metric = Metric(true=data_test.labels.reshape(-1), pred=data_test_pred.labels.reshape(-1))
	test_acc=metric.accuracy()
	test_disp=metric.positive_disparity(s=s_test)
	# print('Testing Acc.:', test_acc)

	data_test_attacked = inproc.attack(data_test, method=method, use_label=False)
	data_test_attacked_pred = inproc.predict(data_test_attacked)
	metric = Metric(true=data_test_attacked.labels.reshape(-1), pred=data_test_attacked_pred.labels.reshape(-1))
	test_adv_acc=metric.accuracy()
	test_adv_disp=metric.positive_disparity(s=s_test)
	# print('Testing Atk Acc.:', test_adv_acc)

	sess.close()
	tf.reset_default_graph()

	report={
		'train_acc':train_acc,
		'train_disp':train_disp,
		'test_acc':test_acc,
		'test_disp':test_disp,
		'test_adv_acc':test_adv_acc,
		'test_adv_disp':test_adv_disp,
	}

	return report

def data_generator():
	for data in ['adult', 'compas']:
		reset_seed(seed)
		dataset, privileged_groups, unprivileged_groups = load_data(data, 'race')
		data_train, data_test = dataset.split([0.7], shuffle=True)
		with gzip.open(f'./data/{data}_train.pkl.gz', 'wb') as handle:
			pickle.dump(data_train, handle)
		df = pd.DataFrame(np.hstack([data_train.features,data_train.labels]), columns=data_train.feature_names+data_train.label_names)
		df.to_csv(f'./data/{data}_train.csv',index=False)
		with gzip.open(f'./data/{data}_test.pkl.gz', 'wb') as handle:
			pickle.dump(data_test, handle)
		df = pd.DataFrame(np.hstack([data_test.features,data_test.labels]), columns=data_test.feature_names+data_test.label_names)
		df.to_csv(f'./data/{data}_test.csv',index=False)
	exit()


if __name__=='__main__':

	import sys, json, time

	params={
		'fairness_reweighing': [None],
		'fairness_disparate': [1.0],
		'fairness_adversarial': [0.1],
	}

	if len(sys.argv)>2:
		data = sys.argv[1].lower().strip()
		attr = sys.argv[2].lower().strip()
		method = sys.argv[3]
		func = locals()[sys.argv[4]]
		pidx = int(sys.argv[5])
		p = params[func.__name__][pidx]
	else:
		data = 'compas'
		attr = 'race'
		method = 'FGSM'
		func = fairness_adversarial
		pidx = 0
		p = params[func.__name__][pidx]

	seed=int(time.time())
	print('Seed:',seed)
	print(data, attr, method, func.__name__, p)

	reset_seed(seed)
	Rf=func(data,attr,method=method,mitigation=True,param=p)
	reset_seed(seed)
	Ro=func(data,attr,method=method,mitigation=False,param=p)

	report={
		'data':data,
		'attr':attr,
		'method':method,
		'func':func.__name__,
		'result_orig':Ro,
		'result_fair':Rf,
	}

	f=open('./result/existings/F2R.txt','a')
	f.write(json.dumps(report)+'\n')
	f.close()



