import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import load_split

for data in ['adult','compas']:
	for attr in ['race','sex']:

		train, test = load_split(data,attr,ret_col=True,del_s=False)
		column=train['name']

		model = LogisticRegression(max_iter=1000)
		model.fit(train['X'], train['y'])

		imp = abs(model.coef_.reshape(-1))
		imp /= sum(imp)
		imp *= imp.shape[0]

		s=test['X'][:,0]

		print('%s %s: %.06f'%(data, attr, round(imp[column.index(attr)],6)))
		print((s==0).sum(), round((s==0).sum()/s.shape[0],4))
		print((s==1).sum(), round((s==1).sum()/s.shape[0],4))
		print()