from copy import deepcopy
import hashlib,random,time,sys

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

import keras
from keras.models import Model,Sequential
from keras.layers import Dense,Activation,Input
from art.classifiers import KerasClassifier as ART_NN

from art.attacks.evasion import ProjectedGradientDescent as ProjectedGradientDescentAttack

global global_suffix
global_suffix='_nS'

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
		print('No such dataset & group option:', dataset_name, protected_attribute_name)
		exit()

	return dataset_original,protected_attribute_set[protected_attribute_name][0],protected_attribute_set[protected_attribute_name][1],optim_options

def get_fair(dataname='german',ratio=0.7,attr='race',transform='OP',no_sensitive=False):

	dataset_orig,privileged_groups,unprivileged_groups,optim_options = LoadData(dataset_name=dataname,protected_attribute_name=attr,raw=False)

	sensitive_index=dataset_orig.feature_names.index(attr)

	dataset_original_train, dataset_original_test=dataset_orig.split([ratio], shuffle=True)

	X_train=dataset_original_train.features.astype(int)
	Y_train=dataset_original_train.labels.reshape(-1).astype(int)
	W_train=dataset_original_train.instance_weights.reshape(-1)

	X_test=dataset_original_test.features.astype(int)
	Y_test=dataset_original_test.labels.reshape(-1).astype(int)
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
	W_train_fair=dataset_fair_train.instance_weights.reshape(-1)

	if no_sensitive:
		X_train[:,sensitive_index]=0
		X_test[:,sensitive_index]=0
		X_train_fair[:,sensitive_index]=0

	feature_names=dataset_fair_train.feature_names
	label_names=dataset_fair_train.label_names

	return feature_names,label_names,X_train_fair,Y_train_fair,W_train_fair,X_train,Y_train,W_train,X_test,Y_test,W_test



def best_acc(proba,true):
	ret=None
	acc=0.0
	thrd=None
	for i in range(1,100):
		threshold=i*0.01
		pred=(proba[:,1]>threshold).astype(int)
		now=(pred==true).sum()/len(true)
		if now>acc:
			acc=now
			ret=pred
			thrd=threshold
	print(thrd)
	return pred

def evaluation(train,test,name,tags=None,model_name='LR',raw_result=False):

	if model_name=='LR':
		clf=LogisticRegression()
		art=ART_LR(clf)

	if raw_result:
		outfile='test_predcition_%s_%s_%s_%s%s.txt'%(tags['dataset'],tags['attribute'],tags['model'],tags['fairness'],global_suffix)
		outdata={}

	art.fit(train[0],train[1],sample_weight=train[2])
	pred=art.predict(test[0])
	if raw_result:
		outdata['original']=pred.tolist()
	pred=(pred[:,1]>0.5).astype(int)
	acc_original=(pred==test[1]).sum()/len(pred)
	print('Accuracy :',acc_original)

	attack=ProjectedGradientDescentAttack(classifier=art)
	X_test_adv=attack.generate(x=test[0])
	pred_adv=art.predict(X_test_adv)
	if raw_result:
		outdata['adversial']=pred_adv.tolist()
	pred_adv=(pred_adv[:,1]>0.5).astype(int)
	acc_attack=(pred_adv==test[1]).sum()/len(pred_adv)
	print('Accuracy_attack :',acc_attack)

	if raw_result:
		f=open('./raw_result/'+outfile,'a')
		f.write(str(outdata)+'\n')
		f.write(str(test[1].tolist())+'\n')
		f.close()

	if not raw_result:
		return acc_original,acc_attack
	else:
		return None,None


if __name__=='__main__':

	# datas=['compas','german','adult']
	# attrs={'adult':['race','sex'],'compas':['race','sex'],'german':['sex','age']}

	# for t in range(0,23):
	# 	for data in datas:
	# 		for attr in attrs[data]:

	if len(sys.argv)<3:
		exit()
	
	data=sys.argv[1]
	attr=sys.argv[2]

	print(data,attr)

	feature_names,label_names,X_train_fair,Y_train_fair,W_train_fair,X_train,Y_train,W_train,X_test,Y_test,W_test=get_fair(
		dataname=data,
		attr=attr,
		ratio=0.7,
		no_sensitive=True
	)

	acc_original,acc_attack=evaluation(
		train=(X_train,Y_train,W_train),
		test=(X_test,Y_test,W_test),
		name=(feature_names,label_names),
		model_name='LR',
		raw_result=True,
		tags={'dataset':data,'attribute':attr,'model':'LogisticRegression','fairness':'OP'}
	)
	acc_original_fair,acc_attack_fair=evaluation(
		train=(X_train_fair,Y_train_fair,W_train_fair),
		test=(X_test,Y_test,W_test),
		name=(feature_names,label_names),
		model_name='LR',
		raw_result=True,
		tags={'dataset':data,'attribute':attr,'model':'LogisticRegression','fairness':'OP'}
	)

	if acc_original!=None:
		print()
		f=open('result_%s_%s%s.txt'%(data,attr,global_suffix),'a')
		f.write(str([acc_original,acc_attack,acc_original_fair,acc_attack_fair])+'\n')
		f.close()
