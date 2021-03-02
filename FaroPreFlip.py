import torch
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from pandas import DataFrame
from metric import Metric
from utils import get_data

from matplotlib import pyplot as plt


class TorchLogistic(torch.nn.Module):
	def __init__(self, inps, bias=True, seed=None):
		super(TorchLogistic, self).__init__()
		if seed is not None:
			torch.manual_seed(seed)
		self.linear_model = torch.nn.Linear(inps, 1, bias=bias)
		self.activation = torch.nn.Sigmoid()
	def forward(self, x):
		return self.activation(self.linear_model(x))

class PreProcFlip(object):
	def __init__(self, data, attr, k=10, seed=24, delta=0.01, kind='SP', credit_lim=1, max_iter=1000):
		self._delta=delta
		self._data=data
		self._attr=attr
		self._k=k
		self._seed=seed
		self._kind=kind
		self._credit_limit=credit_lim
		self._max_iter=max_iter
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

		X,y=get_data(data,attr)
		self._orig_s=X[:,0].reshape(-1)
		self._orig_X=X[:,1:]
		self._orig_y=y.reshape(-1)

		X, X_test, s, s_test, y, y_test=train_test_split(self._orig_X,self._orig_s,self._orig_y,test_size=0.33)

		self._train_pack_np=(X,s,y,y.copy())
		self._test_pack_np=(X_test,s_test,y_test)

		X=torch.tensor(X, dtype=torch.float)
		s=torch.tensor(s.reshape(-1,1), dtype=torch.float)
		y_prime=torch.tensor(y.reshape(-1,1), dtype=torch.float)
		y=torch.tensor(y.reshape(-1,1), requires_grad=True ,dtype=torch.float)

		X_test=torch.tensor(X_test, dtype=torch.float)
		y_test=torch.tensor(y_test.reshape(-1,1), dtype=torch.float)
		
		self._train_pack=(X,s,y,y_prime)
		self._test_pack=(X_test,s_test,y_test)

	def _BCELoss(self,y_pred,y):
		return -torch.mean(y*torch.log(0.999 * y_pred)+(1.0-y)*torch.log(1.0-0.999 * y_pred))
	def _DISPLoss(self,c,y_pred):
		return torch.square(torch.sum(c*y_pred))
	def _AccDisp(self,y,y_pred,s):
		metric=Metric(true=y.reshape(-1).tolist(),pred=y_pred.reshape(-1).tolist())
		return metric.accuracy(), metric.positive_disparity(s=s)


	def fit_transform(self, test_output=True):
		
		X,s,y,y_prime=self._train_pack
		X_np,s_np,y_np,y_np_prime=self._train_pack_np
		X_test,s_test,y_test=self._test_pack

		res={
			'setting': '%s_%s_%s'%(self._kind,self._data,self._attr),
			'train':{
				'acc':[],
				'disp':[],
			},
			'test':{
				'acc':[],
				'disp':[],
			},
			'iter':[],
		}

		model=TorchLogistic(inps=X.shape[1], seed=self._seed)
		optim=torch.optim.Adam(model.parameters(),lr=0.1)
		loss_func=torch.nn.BCELoss()
		credit={}
		
		for it in range(0, self._max_iter):

			# BEGIN: Train modified model
			for epoch in range(0,300):
				optim.zero_grad()
				y_pred=model(X)
				loss=self._BCELoss(y_pred,y_prime)
				loss.backward()
				optim.step()
			acc, disp = self._AccDisp(y_prime, y_pred,s_np)
			if disp<=self._delta:
				break
			res['train']['acc'].append(acc)
			res['train']['disp'].append(disp)
			if test_output:
				print('Iter: %d, Train: (%.4f, %.4f)'%(it+1, acc, disp), end=', ')
			else:
				print('Iter: %d, Train: (%.4f, %.4f)'%(it+1, acc, disp))
			# END

			# BEGIN: Testing
			if test_output:
				y_test_pred = model(X_test)
				acc, disp = self._AccDisp(y_test, y_test_pred, self._test_pack_np[1])
				res['test']['acc'].append(acc)
				res['test']['disp'].append(disp)
				print('Test: (%.4f, %.4f)'%(acc, disp))
			# END: Testing

			res['iter'].append(it+1)

			# BEGIN: Influences
			y.grad=None
			y_pred = model(X)
			loss=self._BCELoss(y_pred,y)
			loss.backward()
			y_pred = y_pred.detach().numpy()
			y_grad = y.grad.numpy().reshape(-1)
			influence_to_utility = - y_grad * np.sign(y_np-0.5) # Condtion 1: Utility
			metric = Metric(pred=y_pred,true=y_np_prime)
			disp = metric.positive_disparity(s=s_np, absolute=False)
			influence_to_parity = np.sign(disp*(s_np-0.5)*(y_np_prime-0.5))
			# END: Influences

			# BEGIN: Flipping
			influence=np.column_stack((influence_to_parity, influence_to_utility, list(range(0,influence_to_utility.shape[0]))))
			influence=influence.tolist()
			influence.sort()

			rest=self._k
			for item in influence:
				if item[0]>0: # No flipping is beneficial to parity
					break
				i=int(item[2])
				if i not in credit:
					credit[i]=self._credit_limit
				elif credit[i]==0:
					continue
				credit[i]-=1
				y_prime[i][0]=1-y_prime[i][0]
				y_np_prime[i]=1-y_np_prime[i]
				rest-=1
				if rest==0: # Successfully picked k flippings
					break

			if rest==self._k: # No eligible flipping is available
				break
			# END: Flipping

		return res

			

def draw(res):

	fig,ax=plt.subplots(1,2)
	fig.set_size_inches(12.8,4.8)

	data=np.array(res['train']['acc']).reshape(-1,1)
	df_train=DataFrame(data, index=res['iter'], columns=['Train Acc.'])
	main_ax=df_train.plot(ax=ax[0])
	main_ax.set_xlabel('Training epochs')
	main_ax.set_ylabel('Accuracy')
	main_ax.set_title(res['setting']+'_train')

	data=np.array(res['train']['disp']).reshape(-1,1)
	df_train=DataFrame(data, index=res['iter'], columns=['Train Disp.'])
	m2nd_ax=df_train.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel('Statistical parity')

	data=np.array(res['test']['acc']).reshape(-1,1)
	df_train=DataFrame(data, index=res['iter'], columns=['Test Acc.'])
	main_ax=df_train.plot(ax=ax[1])
	main_ax.set_xlabel('Training epochs')
	main_ax.set_ylabel('Accuracy')
	main_ax.set_title(res['setting']+'_test')

	data=np.array(res['test']['disp']).reshape(-1,1)
	df_train=DataFrame(data, index=res['iter'], columns=['Test Disp.'])
	m2nd_ax=df_train.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel('Statistical parity')

	fig.tight_layout()



if __name__=='__main__':
	model = PreProcFlip('adult','race',delta=0.09)
	res=model.fit_transform()
	draw(res)
