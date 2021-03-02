import numpy as np
from utils import get_data
from scipy.stats import pearsonr as pearson

for data in ['adult', 'compas']:
	for attr in ['race', 'sex']:
		X, y=get_data(data, attr)
		X=np.hstack([X,y])
		m=X.shape[1]
		print(data, attr, 'sensitive')
		for j in range(0,m):
			print('%.4f'%(pearson(X[:,0],X[:,j]))[0], end='\t')
			if (j+1)%10==0:
				print()
		print()
		print(data, attr, 'label')
		for j in range(0,m):
			print('%.4f'%(pearson(X[:,-1],X[:,j]))[0], end='\t')
			if (j+1)%10==0:
				print()
		print()