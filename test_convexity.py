import numpy as np
from matplotlib import pyplot as plt
from math import log

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

n=100
m=2
s=(np.random.uniform(0,1,n)<0.3).astype(int)
x=np.random.uniform(0,1,(n,m))
y=np.random.uniform(0,2,n).astype(int)

def f(x,s,w,construction=False):
	t=sigmoid(np.dot(x,w))
	u0=np.sum(t*(1-s))
	v0=np.sum(1-s)
	u1=np.sum(t*s)
	v1=np.sum(s)
	p0=u0/v0
	p1=u1/v1
	epsilon=0.01
	if not construction:
		return abs(p0-p1)
	else:
		return (np.exp(epsilon*abs(p0-p1))-1)/epsilon

w=[]
for i in range(-30,31):
	for j in range(-30,31):
		w.append([i,j])
w=np.array(w)
v=[f(x,s,w_) for w_ in w]
z=[f(x,s,w_,construction=True) for w_ in w]

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(w[:,0], w[:,1], v, alpha=0.3, label='MAE')
ax.scatter(w[:,0], w[:,1], z, alpha=0.3, label='BCE')
ax.legend()
plt.show()
