import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gzip
import torch
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy

from aif360.datasets import AdultDataset, CompasDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from ExistingApproaches.disparate_impact_remover import DisparateImpactRemover

from TorchAttackable import TorchNeuralNetworks as TorchAttackable
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

	return data_train, data_test, privileged_groups, unprivileged_groups

def augment(dataset, advexp, ratio=1.0):

	n = int(advexp['X'].shape[0]*ratio)

	ret=dataset.copy()

	ret.features=np.vstack([ret.features, advexp['X'][:n]])
	ret.labels=np.vstack([ret.labels, advexp['y'][:n].reshape(-1,1).astype(float)])
	ret.instance_names=[str(i) for i in range(0,ret.features.shape[0])]
	ret.instance_weights=np.ones(ret.features.shape[0])
	# frame.metadata=None
	ret.protected_attributes=np.vstack([ret.protected_attributes, np.hstack([advexp['s_race'][:n].reshape(-1,1), advexp['s_sex'][:n].reshape(-1,1)])])
	ret.scores=np.stack([ret.scores, advexp['y'][:n].reshape(-1,1).astype(float)])

	return ret

def get_Xsy(data):

	X = data.features
	y = data.labels.reshape(-1)
	s_race_index=data.feature_names.index('race')
	s_race=data.features[:,s_race_index].reshape(-1)
	s_sex_index=data.feature_names.index('sex')
	s_sex=data.features[:,s_sex_index].reshape(-1)
	weight=data.instance_weights.reshape(-1)
	return {
		'X': X,
		'y': deepcopy(y),
		's_race':deepcopy(s_race),
		's_sex':deepcopy(s_sex)
	}

def Arena(data, attr, method='FGSM', wR=0.0, wF=0.0):

	traind, testd, privileged_groups, unprivileged_groups = load_data('adult', 'race')

	train=get_Xsy(traind)
	test=get_Xsy(testd)

	model = TorchAttackable(lr=0.01, n_epoch=500, hiddens=[128], seed=global_seed)
	if attr=='race':
		model.fit(X=train['X'],s=train['s_race'],y=train['y'])
	elif attr=='sex':
		model.fit(X=train['X'],s=train['s_sex'],y=train['y'])

	X_adv = model.AdvExp(X=train['X'],y=train['y'],method=method).detach().numpy()

	advexp = {
		'X': X_adv,
		'y': deepcopy(train['y']),
		's_race':deepcopy(train['s_race']),
		's_sex':deepcopy(train['s_sex'])
	}

	print('Original result:', model.metrics(X=test['X'], y=test['y']))
	print('Attacked result:', model.metrics_attack(X=test['X'], y=test['y'], method=method, use_y=False))
	# ----------------

	train_def = get_Xsy(augment(traind, advexp))

	# print(train_def['X'][0])
	# print(train_def['X'][traind.features.shape[0]])

	print(train_def['X'].shape)

	model = TorchAttackable(lr=0.01, n_epoch=500, hiddens=[128], seed=global_seed)
	if attr=='race':
		model.fit(X=train_def['X'],s=train_def['s_race'],y=train_def['y'])
	elif attr=='sex':
		model.fit(X=train_def['X'],s=train_def['s_sex'],y=train_def['y'])

	print('Defended result train:', model.metrics_attack(X=train['X'], y=train['y'], method=method))
	print('Defended result test:', model.metrics_attack(X=test['X'], y=test['y'], method=method, use_y=False))







if __name__=='__main__':
	Arena('adult','race')