import os, pickle
import pandas as pd

orig = pd.read_csv('../../data/adult_train.csv').to_numpy()
n = orig.shape[0]
print(n)

fnames = os.listdir('.')
for fname in fnames:
	if 'adult' in fname and '.pre' in fname:
		f=open(fname,'rb')
		a=pickle.load(f)
		f.close()
	dR = a['dR']
	dF = a['dF']
	inc = a['downstream_train']['X'].shape[0] - n
	print(dR, dF, inc, n)