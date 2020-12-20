import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
from copy import deepcopy
import hashlib,random,time,sys
import numpy as np
from metric import Metric
from copy import deepcopy

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
	import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
	import get_distortion_adult, get_distortion_german, get_distortion_compas

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from tools.opt_tools import OptTools

from aif360.datasets import StandardDataset

def get_distortion_hospital(vold, vnew):
	return abs(vold-vnew)

def load_hospital():
	df=pd.read_csv('./data/hospital.csv')
	return StandardDataset(
		df=df,
		label_name='label', 
		favorable_classes=[1],
		protected_attribute_names=['sex','race'],
		privileged_classes=[[1.0],[1.0]],
		instance_weights_name=None,
		scores_name='',
		categorical_features=[],
		features_to_keep=list(df.columns),
		features_to_drop=[],
		na_values=[],
		custom_preprocessing=None,
		metadata={
			'label_maps': [{1.0: 'heart', 0.0: 'breath'}],
			'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'},
										 {1.0: 'White', 0.0: 'Non-white'}]
		},
	)

def LoadData(dataset_name,protected_attribute_name,raw=True):

	optim_options=None

	if dataset_name == "adult":
		if raw:
			dataset_original = AdultDataset()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 1}]
			unprivileged_groups = [{'sex': 0}]
			if not raw:
				dataset_original = load_preproc_data_adult(['sex'])
			optim_options = {
				"distortion_fun": get_distortion_adult,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
			if not raw:
				dataset_original = load_preproc_data_adult(['race'])
			optim_options = {
			"distortion_fun": get_distortion_adult,
			"epsilon": 0.05,
			"clist": [0.99, 1.99, 2.99],
			"dlist": [.1, 0.05, 0]
		}
	elif dataset_name == "german":
		if raw:
			dataset_original = GermanDataset()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 1}]
			unprivileged_groups = [{'sex': 0}]
			if not raw:
				dataset_original = load_preproc_data_german(['sex'])
			optim_options = {
				"distortion_fun": get_distortion_german,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "age":
			privileged_groups = [{'age': 1}]
			unprivileged_groups = [{'age': 0}]
			if not raw:
				dataset_original = load_preproc_data_german(['age'])
			optim_options = {
				"distortion_fun": get_distortion_german,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		dataset_original.labels = 2 - dataset_original.labels
		dataset_original.unfavorable_label = 0.
	elif dataset_name == "compas":
		if raw:
			dataset_original = CompasDataset()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 0}]
			unprivileged_groups = [{'sex': 1}]
			if not raw:
				dataset_original = load_preproc_data_compas(['sex'])
			optim_options = {
				"distortion_fun": get_distortion_compas,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
			if not raw:
				dataset_original = load_preproc_data_compas(['race'])
			optim_options = {
				"distortion_fun": get_distortion_compas,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
	elif dataset_name == 'hospital':
		dataset_original = load_hospital()
		if protected_attribute_name == "sex":
			privileged_groups = [{'sex': 0}]
			unprivileged_groups = [{'sex': 1}]
			optim_options = {
				"distortion_fun": get_distortion_hospital,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}
		elif protected_attribute_name == "race":
			privileged_groups = [{'race': 1}]
			unprivileged_groups = [{'race': 0}]
			optim_options = {
				"distortion_fun": get_distortion_hospital,
				"epsilon": 0.05,
				"clist": [0.99, 1.99, 2.99],
				"dlist": [.1, 0.05, 0]
			}

	protected_attribute_set={
		'sex':[[{'sex': 1}],[{'sex': 0}]],
		'age':[[{'age': 1}],[{'age': 0}]],
		'race':[[{'race': 1}],[{'race': 0}]]
	}

	if optim_options==None:
		raise ValueError(f'No such dataset & group option: {dataset_name}, {protected_attribute_name}')

	return dataset_original,protected_attribute_set[protected_attribute_name][0],protected_attribute_set[protected_attribute_name][1],optim_options

def preproc(dataname='adult',ratio=0.7,attr='race',transform='OP',no_sensitive=True):

	dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name=dataname,protected_attribute_name=attr,raw=False)

	sensitive_index=dataset_orig.feature_names.index(attr)

	dataset_original_train, dataset_original_test=dataset_orig.split([ratio], shuffle=True)

	X_train=dataset_original_train.features.astype(int)
	Y_train=dataset_original_train.labels.reshape(-1).astype(int)
	S_train=dataset_original_train.features[:,sensitive_index].reshape(-1)
	W_train=dataset_original_train.instance_weights.reshape(-1)

	X_test=dataset_original_test.features.astype(int)
	Y_test=dataset_original_test.labels.reshape(-1).astype(int)
	S_test=dataset_original_test.features[:,sensitive_index].reshape(-1)
	W_test=dataset_original_test.instance_weights.reshape(-1)

	# Begin pre-processing
	if transform=='RW':
		proc=Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
		dataset_fair_train=proc.fit_transform(dataset_original_train)
	if transform=='OP':
		proc=OptimPreproc(OptTools,optim_options)
		proc=proc.fit(dataset_original_train)
		dataset_fair_train=proc.transform(dataset_original_train,transform_Y=True)
	# End pre-processing

	X_train_fair=dataset_fair_train.features.astype(int)
	Y_train_fair=dataset_fair_train.labels.reshape(-1).astype(int)
	S_train_fair=dataset_fair_train.features[:,sensitive_index].reshape(-1)
	W_train_fair=dataset_fair_train.instance_weights.reshape(-1)

	if no_sensitive:
		X_train=np.delete(X_train,sensitive_index,axis=1)
		X_test=np.delete(X_test,sensitive_index,axis=1)
		X_train_fair=np.delete(X_train_fair,sensitive_index,axis=1)

	feature_names=dataset_fair_train.feature_names
	label_names=dataset_fair_train.label_names

	# os.system('clear')

	train={
		'X':X_train,
		'y':Y_train,
		's':S_train,
	}
	ftrain={
		'X':X_train_fair,
		'y':Y_train_fair,
		's':S_train
	}
	test={
		'X':X_test,
		'y':Y_test,
		's':S_test,
	}

	return train, ftrain, test

if __name__=='__main__':

	train, ftrain, test = preproc(dataname='adult',ratio=0.7,attr='race',transform='OP',no_sensitive=True)
	print((train['X']==train['X_fair']).sum(),train['X'].shape[0]*train['X'].shape[1])


