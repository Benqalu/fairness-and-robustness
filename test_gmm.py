import numpy as np
from sklearn.mixture import GaussianMixture as GMM

from utils import get_data


X,y=get_data('adult','race')
s=X[:,0]
X=X[:,1:]
y=y.reshape(-1)

def getgroup(g):
	return X[s==g],y[s==g]



X_=getgroup(g=0)

model=GMM(n_components=5)
model.fit(X)

print(model.means_)