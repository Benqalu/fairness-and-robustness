import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from art.attacks.evasion import FastGradientMethod as FastGradientMethodAttack
from art.classifiers.scikitlearn import ScikitlearnLogisticRegression as ART_SKLR

from utils import preproc
from metric import Metric


def to_categorical(a):
	return OneHotEncoder().fit_transform(np.array(a).reshape(-1,1)).toarray()

def core(train, test, eps, f):

	model=ART_SKLR(LogisticRegression())
	model.fit(train['X'], to_categorical(train['y']))

	test['y_pred']=model.predict(test['X'])[:,1]
	metric=Metric(true=test['y'], pred=test['y_pred'])
	print('Testing acc. : %.4f'%(metric.accuracy()))
	print('Testing disp.: %.4f'%(metric.positive_disparity(s=test['s'])))
	if f is not None:
		f.write('Testing acc. : %.4f'%(metric.accuracy())+'\n')
		f.write('Testing disp.: %.4f'%(metric.positive_disparity(s=test['s']))+'\n')

	attack=FastGradientMethodAttack(estimator=model, norm=np.inf, eps=eps)
	train['X_adv']=attack.generate(x=train['X'])
	test['X_adv']=attack.generate(x=test['X'])
	test['y_adv_pred']=model.predict(test['X_adv'])[:,1]
	metric=Metric(true=test['y'], pred=test['y_adv_pred'])
	print('Attacked acc. : %.4f'%(metric.accuracy()))
	print('Attacked disp.: %.4f'%(metric.positive_disparity(s=test['s'])))
	if f is not None:
		f.write('Attacked acc. : %.4f'%(metric.accuracy())+'\n')
		f.write('Attacked disp.: %.4f'%(metric.positive_disparity(s=test['s']))+'\n')

	model=ART_SKLR(LogisticRegression())
	model.fit(np.concatenate([train['X'], train['X_adv']]), to_categorical(np.concatenate([train['y'], train['y']])))
	
	test['y_pred']=model.predict(test['X'])[:,1]
	metric=Metric(true=test['y'], pred=test['y_pred'])
	print('Def testing acc. : %.4f'%(metric.accuracy()))
	print('Def testing disp.: %.4f'%(metric.positive_disparity(s=test['s'])))
	if f is not None:
		f.write('Def testing acc. : %.4f'%(metric.accuracy())+'\n')
		f.write('Def testing disp.: %.4f'%(metric.positive_disparity(s=test['s']))+'\n')

	test['y_adv_pred']=model.predict(test['X_adv'])[:,1]
	metric=Metric(true=test['y'], pred=test['y_adv_pred'])
	print('Def attacked acc. : %.4f'%(metric.accuracy()))
	print('Def attacked disp.: %.4f'%(metric.positive_disparity(s=test['s'])))
	if f is not None:
		f.write('Def attacked acc. : %.4f'%(metric.accuracy())+'\n')
		f.write('Def attacked disp.: %.4f'%(metric.positive_disparity(s=test['s']))+'\n')

def attack(data='adult', attr='race', eps=0.1, f=None):
	report={}

	train, ftrain, test = preproc(dataname=data,ratio=0.7,attr=attr,transform='OP',no_sensitive=True)

	print('****** Original dataset ******')
	if f is not None:
		f.write('****** Original dataset ******\n')
	core(train=deepcopy(train), test=deepcopy(test), eps=eps, f=f)

	print('****** Fairness-mitigated dataset ******')
	if f is not None:
		f.write('****** Fairness-mitigated dataset ******\n')
	core(train=deepcopy(ftrain), test=deepcopy(test), eps=eps, f=f)

	report={
		'train':train,
		'ftrain':ftrain,
		'test':test,
	}

if __name__=='__main__':
	data='compas'
	attr='sex'
	for data in ['compas', 'adult']:
		for attr in ['sex', 'race']:
			f=open('result/result_%s_%s.txt'%(data,attr), 'w')
			attack(data=data, attr=attr, eps=0.1, f=f)
			f.close()