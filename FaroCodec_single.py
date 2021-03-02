import torch, sys
import numpy as np
from sklearn.linear_model import LogisticRegression


class Discriminator(torch.nn.Module):
	def __init__(self, hiddens, inps):
		super().__init__()
		struct = [inps]+hiddens+[1]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i], bias=True))
			if i!=len(struct)-1:
				layers.append(torch.nn.ReLU())
			else:
				layers.append(torch.nn.Sigmoid())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		return output

class AutoCodec(torch.nn.Module):
	def __init__(self, hiddens, inps, outs, out_activation=None):
		super().__init__()
		struct = [inps]+hiddens+[outs]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i],bias=True))
			if i!=len(struct)-1:
				layers.append(torch.nn.ReLU())
			else:
				if out_activation is not None:
					layers.append(out_activation())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		return output

class FaroCodec(object):

	def __init__(self, seed=None, codec_hiddens=[32], disc_hiddens=[], wF=0.0, wR=0.0, codec_lr=0.01, disc_lr=0.01, n_epoch=1000, batch_size=None, interval=1):
		self._wF=wF
		self._wR=wR
		self._codec_lr=codec_lr
		self._disc_lr=disc_lr
		self._n_epoch=n_epoch
		self._codec_hiddens=codec_hiddens
		self._disc_hiddens=disc_hiddens
		self._BCE_loss=torch.nn.BCELoss()
		self._batch_size=batch_size
		self._interval=interval
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)

	def fit(self,X,y):

		self._np_sX=X
		self._np_s=X[:,0].reshape(-1)
		self._np_X=X[:,1:]
		self._np_y=y.reshape(-1)

		if self._batch_size is None:
			self._batch_size=X.shape[0]

		s = torch.tensor(self._np_s.reshape(-1,1), dtype=torch.float, requires_grad=False)
		X = torch.tensor(self._np_X, dtype=torch.float, requires_grad=False)
		y = torch.tensor(self._np_y.reshape(-1,1), dtype=torch.float, requires_grad=False)
		D = torch.tensor(np.hstack([self._np_X,self._np_y.reshape(-1,1)]), dtype=torch.float, requires_grad=False)

		s_confused=torch.ones((s.shape[0], 1))*0.5

		model_codec=AutoCodec(hiddens=self._codec_hiddens, inps=D.shape[1], outs=D.shape[1]-1, out_activation=torch.nn.Sigmoid)
		optim_codec=torch.optim.Adam(model_codec.parameters(), lr=self._codec_lr, weight_decay=0.01)

		model_disc=Discriminator(hiddens=self._disc_hiddens, inps=D.shape[1])
		optim_disc=torch.optim.Adam(model_disc.parameters(), lr=self._disc_lr, weight_decay=0.01)

		all_indices=np.arange(0, X.shape[0])
		for epoch in range(0,self._n_epoch):
			indices=np.random.choice(all_indices, size=self._batch_size, replace=False)
			try:
				optim_codec.zero_grad()
				X_=model_codec(D[indices])
				loss_U=self._BCE_loss(X_,X[indices])
				loss=1.0*loss_U
				D_=torch.hstack([X_,y[indices]])
				s_pred=model_disc(D_)
				loss_F=-self._BCE_loss(s_pred,s_confused[indices])
				loss+=self._wF * loss_F
				loss.backward()
				optim_codec.step()

				optim_disc.zero_grad()
				X_=model_codec(D[indices])
				D_=torch.hstack([X_,y[indices]])
				s_pred=model_disc(D_)
				loss_D=self._BCE_loss(s_pred,s[indices])
				loss_D.backward()
				optim_disc.step()

				# disc_acc=Metric(true=self._np_s.reshape(-1),pred=np.array(s_pred.tolist()).reshape(-1)).accuracy()

				# print('Game:', epoch, 'loss_U=%.6f'%loss_U, end=' ')
				# print('loss_F=%.6f'%loss_F, end=' ')
				# # print('loss_D=%.6f'%loss_D, end='\r')
				# print('Acc_D=%.6f'%disc_acc, end='\r')
				# sys.stdout.flush()

				D=torch.hstack([X, y])
				X_,y_=model_codec(D).tolist(), y.tolist()
				acc,disp=test_disp(X_,y_,self._np_sX,self._np_y)
				print('>>> epoch: %d, Acc.: %.4f, Disp.:%.4f'%(epoch, acc, disp))

			except KeyboardInterrupt:
				break

		print()
		print('>>>', test_disp(X_,y,self._np_sX,self._np_y))

		X_=model_codec(D)
		y_=y
		return X_.tolist(),y_.tolist()

def test_disp(X,y,orig_X,orig_y):
	
	if type(X) is not list:
		X_=np.array(X.tolist())
		y_=np.array(y.tolist()).reshape(-1).round()
	else:
		X_=np.array(X)
		y_=np.array(y).reshape(-1).round()
	model=LogisticRegression(max_iter=1000)
	model.fit(X_,y_)
	orig_y_pred=model.predict(orig_X[:,1:])
	metric=Metric(true=orig_y,pred=orig_y_pred)
	return metric.accuracy(), metric.positive_disparity(s=orig_X[:,0])



if __name__=='__main__':
	from utils import get_data
	from metric import Metric

	X,y=get_data('compas', 'race')

	model=FaroCodec(
		codec_hiddens=[512,512], 
		disc_hiddens=[], 
		wF=0.3, 
		wR=0.0, 
		codec_lr=0.001,
		disc_lr=0.001,
		n_epoch=30000,
		seed=24,
		batch_size=64
	)
	X_,y_=model.fit(X,y)
	y_=np.array(y_).reshape(-1).round()

	# acc,disp=test_disp(X_,y_,X,y)
	# print(acc,disp)
	



