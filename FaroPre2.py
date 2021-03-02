import torch, sys
import numpy as np
from metric import Metric
from utils import get_data
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
				layers.append(torch.nn.ReLU())
			else:
				layers.append(torch.nn.Sigmoid())
		self.model = torch.nn.Sequential(*layers)
	def forward(self, x):
		output = self.model(x)
		return output

def PreProcess(X,s,y):

	orig_X=X
	orig_s=s
	orig_y=y

	bceloss=torch.nn.BCELoss()

	s = torch.tensor(orig_s, dtype=torch.float, requires_grad=False)
	X = torch.tensor(orig_X, dtype=torch.float, requires_grad=True)
	y = torch.tensor(orig_y, dtype=torch.float, requires_grad=True)

	model=Discriminator(inps=X.shape[1]+1, hiddens=[])
	optim=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
	optimX=torch.optim.Adam([X], lr=0.01)

	n_epoch=300

	chosen=set([])

	for _ in range(0,100):

		for epoch in range(0,n_epoch):
			if epoch==n_epoch-2:
				optimX.zero_grad()
			optim.zero_grad()
			Xy = torch.hstack([X,y])
			s_pred=model(Xy)
			loss=bceloss(s_pred,s)
			loss.backward()
			optim.step()
		
		optimX.step()

		if _%10ø==0:
			acc,disp=test_disp(X=np.array(X.tolist()),y=np.array(y.tolist()),orig_X=orig_X,orig_y=orig_y,s=orig_s.reshape(-1))
			print(acc,disp)



def test_disp(X,y,orig_X,orig_y,s):
	if type(X) is not list:
		X_=np.array(X.tolist())
		y_=np.array(y.tolist()).reshape(-1).round()
	else:
		X_=np.array(X)
		y_=np.array(y).reshape(-1).round()
	model=LogisticRegression(max_iter=300)
	model.fit(X_,y_)
	orig_y_pred=model.predict(orig_X)
	metric=Metric(true=orig_y,pred=orig_y_pred)
	return metric.accuracy(), metric.positive_disparity(s=s)

if __name__=='__main__':
	X,y=get_data('compas', 'race')

	s=X[:,0].reshape(-1,1)
	X=X[:,1:]
	y=y.reshape(-1,1)
	
	PreProcess(X,s,y)
