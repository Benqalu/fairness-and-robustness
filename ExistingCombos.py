import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gzip
import torch
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# from aif360.datasets import AdultDataset, CompasDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from ExistingApproaches.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from metric import Metric

seed=None
global_lr=0.1
global_nepoch=500

def reset_seed(seed):
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		tf.random.set_random_seed(seed)

reset_seed(seed)

def disparity(s,y_pred):
	metric = Metric(true=None, pred=y_pred)
	return metric.positive_disparity(s=s)

class TorchNNCore(torch.nn.Module):
	def __init__(self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU):
		super(TorchNNCore, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		struct = [inps]+hiddens+[1]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i], bias=bias))
			if i==len(struct)-1:
				layers.append(torch.nn.Sigmoid())
			else:
				layers.append(hidden_activation())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		return output

class TorchLogisticRegression(object):
	def __init__(self, lr=0.01, n_epoch=500):
		self._lr=lr
		self._n_epoch=n_epoch
		self._model=None
		self._loss_func=torch.nn.BCELoss(reduction='none')
	def fit(self,X,s,y,weight=None):
		X=torch.tensor(X, dtype=torch.float)
		y=torch.tensor(y.reshape(-1,1), dtype=torch.float)
		if weight is not None:
			weight=torch.tensor(weight.reshape(-1,1), dtype=torch.float)
		self._model=TorchNNCore(inps=X.shape[1], hiddens=[128], seed=seed)
		optim=torch.optim.Adam(self._model.parameters(), lr=self._lr)# weight_decay=1E-4)
		for epoch in range(self._n_epoch):
			optim.zero_grad()
			y_pred=self._model(X)
			if weight is not None:
				loss=(self._loss_func(y_pred, y) * weight).mean()
			else:
				loss=self._loss_func(y_pred, y).mean()
			loss.backward()
			optim.step()
	def predict(self,X):
		X_=torch.tensor(X, dtype=torch.float)
		return self._model(X_).detach().numpy().reshape(-1)
	def predict_attack(self,X,y,method='FGSM'):
		if method=='FGSM':
			X_=torch.tensor(X, dtype=torch.float, requires_grad=True)
			y_=torch.tensor(y.reshape(-1,1), dtype=torch.float)
			y_pred=self._model(X_)
			loss=self._loss_func(y_pred,y_).mean()
			noise=torch.sign(torch.autograd.grad(loss,X_)[0])
			return self._model(X_+noise*0.1).detach().numpy().reshape(-1)
		elif method=='PGD':
			X_=torch.tensor(X, dtype=torch.float, requires_grad=True)
			y_=torch.tensor(y.reshape(-1,1), dtype=torch.float)
			for i in range(0,10):
				y_pred=self._model(X_)
				loss=self._loss_func(y_pred,y_).mean()
				noise=torch.sign(torch.autograd.grad(loss,X_)[0])*0.01
				X_=torch.tensor(X_+noise, dtype=torch.float, requires_grad=True)
			return self._model(X_).detach().numpy().reshape(-1)

	def accuracy(self,X,y):
		y_pred=self.predict(X)
		metric=Metric(true=y,pred=y_pred)
		return metric.accuracy()
	def accuracy_attack(self,X,y,method='FGSM'):
		y_pred=self.predict_attack(X,y,method=method)
		metric=Metric(true=y,pred=y_pred)
		return metric.accuracy()
	def disparity(self,X,s,y):
		y_pred=self.predict(X)
		metric=Metric(true=y,pred=y_pred)
		return metric.positive_disparity(s=s)
	def disparity_attack(self,X,s,y,method='FGSM'):
		y_pred=self.predict_attack(X,y,method=method)
		metric=Metric(true=y,pred=y_pred)
		return metric.positive_disparity(s=s)

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

		# df = pd.read_csv('./data/compas.csv')
		# df_np = df.to_numpy()
		# columns=list(df.columns)
		# ret_data.feature_names=columns[:-1]
		# ret_data.label_names=[columns[-1]]
		# ret_data.features=df_np[:,:-1].astype(float)
		# ret_data.labels=df_np[:,-1].reshape(-1,1).astype(float)
		# ret_data.scores=df_np[:,-1].reshape(-1,1).astype(float)
		# sensitive_attribute_index = [columns.index(z) for z in ret_data.protected_attribute_names]
		# ret_data.protected_attributes = ret_data.features[:,sensitive_attribute_index]
		# ret_data.instance_weights = np.ones(ret_data.features.shape[0])
		# ret_data.instance_names = [str(i) for i in range(ret_data.features.shape[0])]

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


def fairness_disparate(data, attr, method='FGSM', mitigation=True, param=1.0):

	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)
	
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	if mitigation:
		preproc = DisparateImpactRemover(repair_level=param,sensitive_attribute=attr)
		data_train_proc = preproc.fit_transform(data_train)
		X,s,y,weight=get_Xsy(data_train_proc,attr,del_s=False)
		model=TorchLogisticRegression(lr=global_lr, n_epoch=global_nepoch)
		model.fit(X,s,y,weight)
		train_acc=model.accuracy(X,y)
		test_acc=model.accuracy(X_test,y_test)
		attack_acc=model.accuracy_attack(X_test,y_test,method)
		test_disp=model.disparity(X_test,s_test,y_test)
		attack_disp=model.disparity_attack(X_test,s_test,y_test,method)
		report={'train_acc':train_acc,'test_acc':test_acc,'attack_acc':attack_acc,'test_disp':test_disp,'attack_disp':attack_disp,'robustness_score':test_acc - attack_acc}
		print('Training Acc.:', train_acc)
		print('Testing Acc.:', test_acc)
		print('Testing Atk Acc.:', attack_acc)
		return report
	else:
		X,s,y,weight=get_Xsy(data_train,attr,del_s=False)
		model=TorchLogisticRegression(lr=global_lr, n_epoch=global_nepoch)
		model.fit(X,s,y)
		train_acc=model.accuracy(X,y)
		test_acc=model.accuracy(X_test,y_test)
		attack_acc=model.accuracy_attack(X_test,y_test,method)
		test_disp=model.disparity(X_test,s_test,y_test)
		attack_disp=model.disparity_attack(X_test,s_test,y_test,method)
		report={'train_acc':train_acc,'test_acc':test_acc,'attack_acc':attack_acc,'test_disp':test_disp,'attack_disp':attack_disp,'robustness_score':test_acc - attack_acc}
		print('Training Acc.:', train_acc)
		print('Testing Acc.:', test_acc)
		print('Testing Atk Acc.:', attack_acc)
		return report

def fairness_reweighing(data, attr, method='FGSM', mitigation=True):
	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)
	
	preproc = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
	data_train_proc = preproc.fit_transform(data_train)

	X,s,y,weight=get_Xsy(data_train_proc,attr,del_s=False)
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	if mitigation:
		model=TorchLogisticRegression(lr=global_lr, n_epoch=global_nepoch)
		model.fit(X,s,y,weight)
		train_acc=model.accuracy(X,y)
		test_acc=model.accuracy(X_test,y_test)
		attack_acc=model.accuracy_attack(X_test,y_test,method)
		test_disp=model.disparity(X_test,s_test,y_test)
		attack_disp=model.disparity_attack(X_test,s_test,y_test,method)
		report={'train_acc':train_acc,'test_acc':test_acc,'attack_acc':attack_acc,'test_disp':test_disp,'attack_disp':attack_disp,'robustness_score':test_acc - attack_acc}
		print('Training Acc.:', train_acc)
		print('Testing Acc.:', test_acc)
		print('Testing Atk Acc.:', attack_acc)
		return report
	else:
		model=TorchLogisticRegression(lr=global_lr, n_epoch=global_nepoch)
		model.fit(X,s,y)
		train_acc=model.accuracy(X,y)
		test_acc=model.accuracy(X_test,y_test)
		attack_acc=model.accuracy_attack(X_test,y_test,method)
		test_disp=model.disparity(X_test,s_test,y_test)
		attack_disp=model.disparity_attack(X_test,s_test,y_test,method)
		report={'train_acc':train_acc,'test_acc':test_acc,'attack_acc':attack_acc,'test_disp':test_disp,'attack_disp':attack_disp,'robustness_score':test_acc - attack_acc}
		print('Training Acc.:', train_acc)
		print('Testing Acc.:', test_acc)
		print('Testing Atk Acc.:', attack_acc)
		return report

def fairness_adversarial(data, attr, method='FGSM', mitigation=True, param=0.1):
	data_train, data_test, privileged_groups, unprivileged_groups = load_data(data,attr)

	scaler=MinMaxScaler()
	data_train.features = scaler.fit_transform(data_train.features)
	data_test.features = scaler.fit_transform(data_test.features)
	X_test,s_test,y_test,_=get_Xsy(data_test,attr,del_s=False)

	sess = tf.Session()
	inproc=AdversarialDebiasing(
		privileged_groups=privileged_groups,
		unprivileged_groups=unprivileged_groups,
		scope_name='debiased_classifier',
		classifier_num_hidden_units=128,
		num_epochs=global_nepoch,
		lr=global_lr,
		debias=mitigation,
		batch_size=data_train.features.shape[0],
		sess=sess,
		adversary_loss_weight=param
	)
	inproc.fit(data_train)

	data_train_pred = inproc.predict(data_train)
	metric = Metric(true=data_train.labels, pred=data_train_pred.labels)
	train_acc=metric.accuracy()
	print('Training Acc.:', train_acc)

	data_test_pred = inproc.predict(data_test)
	metric = Metric(true=data_test.labels, pred=data_test_pred.labels)
	test_acc=metric.accuracy()
	test_disp=metric.positive_disparity(s=s_test)
	print('Testing Acc.:', test_acc)

	data_test_attacked = inproc.attack(data_test, method=method)
	data_test_attacked_pred = inproc.predict(data_test_attacked)
	metric = Metric(true=data_test_attacked.labels, pred=data_test_attacked_pred.labels)
	attack_acc=metric.accuracy()
	attack_disp=metric.positive_disparity(s=s_test)
	print('Testing Atk Acc.:', attack_acc)

	sess.close()
	tf.reset_default_graph()

	report={'train_acc':train_acc,'test_acc':test_acc,'attack_acc':attack_acc,'test_disp':test_disp,'attack_disp':attack_disp,'robustness_score':test_acc - attack_acc}
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

	for func in [fairness_reweighing, fairness_disparate, fairness_adversarial]:
		for method in ['FGSM']:#,'PGD']:
			for data in ['compas']:
				for attr in ['race']:#,'sex']:

					Rf=func(data,attr,method=method,mitigation=True)
					Ro=func(data,attr,method=method,mitigation=False)
					print('>>>',data,attr,round((Ro['robustness_score']-Rf['robustness_score'])/Ro['robustness_score'],4))

					report={
						'data':data,
						'attr':attr,
						'attack':method,
						'mitigation':func.__name__,
						'result_orig':Ro,
						'result_fair':Rf,
						'change':(Ro['robustness_score']-Rf['robustness_score'])/Ro['robustness_score'],
					}

					f=open('./result/existings/r_on_f.txt','a')
					f.write(str(report)+'\n')
					f.close()






