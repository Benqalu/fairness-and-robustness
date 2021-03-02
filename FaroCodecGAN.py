import torch, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True)

class Discriminator(torch.nn.Module):
	def __init__(self, hiddens, inps):
		super().__init__()
		struct = [inps]+hiddens+[1]
		layers = []
		for i in range(1,len(struct)):
			layers.append(torch.nn.Linear(in_features=struct[i-1], out_features=struct[i], bias=True))
			if i!=len(struct)-1:
				layers.append(torch.nn.LeakyReLU())
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
				layers.append(torch.nn.LeakyReLU())
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

		s = torch.tensor(self._np_s.reshape(-1,1), dtype=torch.float, requires_grad=False)
		X = torch.tensor(self._np_X, dtype=torch.float, requires_grad=False)
		y = torch.tensor(self._np_y.reshape(-1,1), dtype=torch.float, requires_grad=False)

		s_confused=torch.ones((s.shape[0], 1))*0.5

		model_codec=AutoCodec(hiddens=self._codec_hiddens, inps=X.shape[1]+1, outs=X.shape[1], out_activation=torch.nn.Sigmoid)
		optim_codec=torch.optim.Adam(model_codec.parameters(), lr=self._codec_lr, weight_decay=0.01)

		model_disc=Discriminator(hiddens=self._disc_hiddens, inps=X.shape[1]+1)
		optim_disc=torch.optim.Adam(model_disc.parameters(), lr=self._disc_lr, weight_decay=0.01)

		all_indices=np.arange(0, X.shape[0])
		for epoch in range(0,self._n_epoch):

			try:

				indices=np.random.choice(all_indices, size=self._batch_size, replace=False)

				for _ in range(0,10):
					# Codec begin
					optim_codec.zero_grad()

					X_true=X[indices]
					y_true=y[indices]
					# s_true=s[indices]
					D_true=torch.hstack([X_true, y_true])
					l_true=torch.ones((len(indices), 1))

					X_fake=model_codec(D_true)
					y_fake=y_true
					# s_fake=s_true
					D_fake=torch.hstack([X_fake, y_fake])
					l_fake=torch.zeros((len(indices), 1))

					loss_codec = self._BCE_loss(model_disc(D_fake), l_true)

					# X_fake_center = torch.mean(X_fake, axis=0)
					# loss_diversity = -torch.mean(torch.abs(X_fake - X_fake_center))
					loss_diversity = torch.mean(torch.abs(X_fake-X_true))

					loss_codec_diversity = loss_codec

					loss_codec_diversity.backward()
					optim_codec.step()
					# Codec end

				# Disc begin
				optim_disc.zero_grad()

				X_true=X[indices]
				y_true=y[indices]
				# s_true=s[indices]
				D_true=torch.hstack([X_true, y_true])
				l_true=torch.ones((len(indices), 1))

				X_fake=model_codec(D_true)
				y_fake=y_true
				# s_fake=s_true
				D_fake=torch.hstack([X_fake, y_fake])
				l_fake=torch.zeros((len(indices), 1))

				loss_disc_true=self._BCE_loss(model_disc(D_true), l_true)
				loss_disc_fake=self._BCE_loss(model_disc(D_fake), l_fake)
				loss_disc=0.5*(loss_disc_true+loss_disc_fake)

				loss_disc.backward()
				optim_disc.step()
				# Disc end

				if epoch%self._interval==0:
					# print('>>> epoch: %d, loss_codec: %.6f, loss_div: %.6f, loss_disc: %.6f'%(epoch, loss_codec.tolist(), loss_diversity.tolist(), loss_disc.tolist()))
					D=torch.hstack([X, y])
					X_,y_=model_codec(D).tolist(), y.tolist()
					acc,disp=test_disp(X_,y_,self._np_sX,self._np_y)
					print('>>> epoch: %d, Acc.: %.4f, Disp.:%.4f'%(epoch, acc, disp))

			except KeyboardInterrupt:
				break

		D=torch.hstack([X, y])
		X_fake=model_codec(D)
		return X_fake.tolist(), y.tolist()


def test_disp(X,y,orig_X,orig_y):
	if type(X) is not list:
		X_=np.array(X.tolist())
		y_=np.array(y.tolist()).reshape(-1).round()
	else:
		X_=np.array(X)
		y_=np.array(y).reshape(-1).round()
	model=LogisticRegression(max_iter=300)
	model.fit(X_,y_)
	orig_y_pred=model.predict(orig_X[:,1:])
	metric=Metric(true=orig_y,pred=orig_y_pred)
	return metric.accuracy(), metric.positive_disparity(s=orig_X[:,0])



if __name__=='__main__':
	from utils import get_data
	from metric import Metric

	X,y=get_data('compas', 'race')

	model=FaroCodec(
		codec_hiddens=[128,1024], 
		disc_hiddens=[], 
		wF=0.0, 
		wR=0.0, 
		codec_lr=0.001,
		disc_lr=0.001,
		n_epoch=30000,
		seed=24,
		batch_size=64,
		interval=10,
	)
	X_,y_=model.fit(X,y)
	y_=np.array(y_).reshape(-1).round()

	print(np.around(X[0], decimals=4))
	print(np.around(X_[0], decimals=4))
	print(np.around(np.array(X_[0])-np.array(X_[1]), decimals=4))
	print(np.around(np.array(X[0])-np.array(X[1]), decimals=4))
	

	acc,disp=test_disp(X_,y_,X,y)
	print(acc,disp)
	



