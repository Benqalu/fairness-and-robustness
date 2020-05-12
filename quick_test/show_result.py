import os
import numpy as np

fnames=os.listdir('.')
for fname in fnames:
	if '.txt' not in fname:
		continue
	data=[]
	f=open(fname)
	for row in f:
		data.append(eval(row))
	f.close()

	data=np.array(data)

	print(fname.split('.')[0][7:],end='\t')
	for item in data.mean(axis=0):
		print(item,end='\t')
	print()