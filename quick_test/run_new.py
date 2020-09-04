from copy import deepcopy
import hashlib,random,time,sys
import numpy as np
from metric import Metric
from copy import deepcopy
import argparse

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from tools.opt_tools import OptTools

from sklearn.ensemble import RandomForestClassifier
from art.classifiers.scikitlearn import ScikitlearnRandomForestClassifier as ART_RF

from sklearn.linear_model import LogisticRegression
from art.classifiers.scikitlearn import ScikitlearnLogisticRegression as ART_LR

# import keras
# from keras.models import Model,Sequential
# from keras.layers import Dense,Activation,Input
# from art.classifiers import KerasClassifier as ART_NN

from art.attacks.evasion import ProjectedGradientDescent as ProjectedGradientDescentAttack

def md5(s):
	a=hashlib.md5()
	a.update(str.encode(str(s)))
	return a.hexdigest()

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

	protected_attribute_set={
		'sex':[[{'sex': 1}],[{'sex': 0}]],
		'age':[[{'age': 1}],[{'age': 0}]],
		'race':[[{'race': 1}],[{'race': 0}]]
	}

	if optim_options==None:
		raise ValueError(f'No such dataset & group option: {dataset_name}, {protected_attribute_name}')

	return dataset_original,protected_attribute_set[protected_attribute_name][0],protected_attribute_set[protected_attribute_name][1],optim_options

def get_fair(dataname='german',ratio=0.7,attr='race',transform='OP',no_sensitive=False):

	if transform=='OP':
		dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name=dataname,protected_attribute_name=attr,raw=False)
	elif transform=='RW':
		dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name=dataname,protected_attribute_name=attr,raw=True)

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

	train_fair={
		'X':X_train_fair,
		'Y':Y_train_fair,
		'S':S_train_fair,
		'W':W_train_fair,
	}
	train={
		'X':X_train,
		'Y':Y_train,
		'S':S_train,
		'W':W_train,
	}
	test={
		'X':X_test,
		'Y':Y_test,
		'S':S_test,
		'W':W_test,
	}

	return feature_names,label_names,train_fair,train,test

def evaluation(train,test,name,tags=None,model_name='LR',raw_result=False):

	if model_name=='LR':
		clf=LogisticRegression(max_iter=1000)
		art=ART_LR(clf)

	if raw_result:
		raw_res_file='test_predcition_%s_%s_%s_%s_%s.txt'%(tags['dataset'],tags['attribute'],tags['model'],tags['fairness'],tags['no_sensitive'])
		raw_res_data={'truth_train':train['Y'].tolist()}
		raw_res_data={'truth_test':test['Y'].tolist()}

	report={'train':{},'test':{},'train_attack':{},'test_attack':{}}


	art.fit(train['X'],train['Y'],sample_weight=train['W'])
	attack=ProjectedGradientDescentAttack(classifier=art)

	# Training
	pred=art.predict(train['X'])
	if raw_result:
		raw_res_data['origin_train']=pred.tolist()
	metric=Metric(true=train['Y'],pred=pred)
	report['train']['accuracy']=metric.accuracy()
	report['train']['recall_disparity']=metric.recall_disparity(train['S'])
	print('TRAIN.accuracy :',report['train']['accuracy'])
	print('TRAIN.recall_disparity:',report['train']['recall_disparity'])

	X_train_adv=attack.generate(x=train['X'])
	pred_adv=art.predict(X_train_adv)
	if raw_result:
		raw_res_data['attack_train']=pred_adv.tolist()
	metric=Metric(true=train['Y'],pred=pred_adv)
	report['train_attack']['accuracy']=metric.accuracy()
	report['train_attack']['recall_disparity']=metric.recall_disparity(train['S'])
	print('TRAIN_attack.accuracy:',report['train_attack']['accuracy'])
	print('TRAIN_attack.recall_disparity:',report['train_attack']['recall_disparity'])




	# Testing
	pred=art.predict(test['X'])
	if raw_result:
		raw_res_data['origin_test']=pred.tolist()
	metric=Metric(true=test['Y'],pred=pred)
	report['test']['accuracy']=metric.accuracy()
	report['test']['recall_disparity']=metric.recall_disparity(test['S'])
	print('TEST.accuracy :',report['test']['accuracy'])
	print('TEST.recall_disparity:',report['test']['recall_disparity'])

	X_test_adv=attack.generate(x=test['X'])
	pred_adv=art.predict(X_test_adv)
	if raw_result:
		raw_res_data['attack_test']=pred_adv.tolist()
	metric=Metric(true=test['Y'],pred=pred_adv)
	report['test_attack']['accuracy']=metric.accuracy()
	report['test_attack']['recall_disparity']=metric.recall_disparity(test['S'])
	print('TEST_attack.accuracy:',report['test_attack']['accuracy'])
	print('TEST_attack.recall_disparity:',report['test_attack']['recall_disparity'])

	# Output
	if raw_result:
		f=open('./raw_result/'+raw_res_file,'a')
		f.write(str(raw_res_data)+'\n')
		f.close()

	return report

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', action='store', default='compas')
	parser.add_argument('-a', action='store', default='sex')
	parser.add_argument('-s', action='store_true')
	parser.add_argument('-f', action='store', default='RW')
	args = parser.parse_args()

	no_sensitive=args.s
	transform=args.f
	data=args.d
	attr=args.a

	if no_sensitive:
		sensitive_suffix='nS'
	else:
		sensitive_suffix=''

	print(vars(args))

	feature_names,label_names,train_fair,train,test=get_fair(
		dataname=data,
		attr=attr,
		ratio=0.7,
		transform=transform,
		no_sensitive=no_sensitive
	)

	res={}

	print('**********Origin**********')
	res['origin']=evaluation(
		train=train,
		test=test,
		name=(feature_names,label_names),
		model_name='LR',
		raw_result=True,
		tags={'dataset':data,'attribute':attr,'model':'LogisticRegression','fairness':transform,'no_sensitive':sensitive_suffix}
	)
	print('**********Faired**********')
	res['faired']=evaluation(
		train=train_fair,
		test=test,
		name=(feature_names,label_names),
		model_name='LR',
		raw_result=True,
		tags={'dataset':data,'attribute':attr,'model':'LogisticRegression','fairness':transform,'no_sensitive':sensitive_suffix}
	)

	f=open('result_%s_%s_%s_%s.txt'%(data,attr,transform,sensitive_suffix),'a')
	f.write(str(res)+'\n')
	f.close()
