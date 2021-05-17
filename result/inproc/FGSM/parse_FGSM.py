import numpy as np
import gzip, json

def angle(data, attr, oR, oF):
	epochs = None
	res = {}
	for i in range(1,12):
		f=gzip.open('./RnF_%d.txt.gz'%(i),'rt')
		for row in f:
			obj=json.loads(row)
			if epochs is None:
				epochs=obj['epoch']
			if (obj['data'], obj['attr']) != (data, attr):
				continue
			wR = obj['wR']
			wF = obj['wF']
			param = (wR, wF)
			if param!=(oR, oF):
				continue
			if param not in res:
				res[param]={
					'data':np.array(obj['angle_rf']),
					'count':1
				}
			else:
				res[param]['data']+=np.array(obj['angle_rf'])
				res[param]['count']+=1
		f.close()
	for param in res:
		res[param]=res[param]['data']/res[param]['count']
		for i in range(0,len(epochs)):
			# if epochs[i]%20==1 or i==len(epochs)-1:
			print((epochs[i], res[param][i]),end=' ')
		print()

def accuracy(data, attr):
	wR_list = [
		0.00,
		0.01,
		0.02,
		0.03,
		0.04,
		0.05,
		0.06,
		0.07,
		0.08,
		0.09,
		0.10,
		0.20,
		0.30,
		0.40,
		0.50,
		0.60,
		0.70,
		0.80,
		0.90,
		1.00,
	]
	wF_list = [round(0.05 * i, 2) for i in range(0, 21)]

	epochs = None
	res = {}
	for i in range(1,12):
		f=gzip.open('./RnF_%d.txt.gz'%(i),'rt')
		for row in f:
			obj=json.loads(row)
			if epochs is None:
				epochs=obj['epoch']
			if (obj['data'], obj['attr']) != (data, attr):
				continue
			wR = obj['wR']
			wF = obj['wF']
			param = (wR, wF)
			if param not in res:
				res[param]={
					'data':np.array([obj['test_metric'][-1][0], obj['test_metric_attack'][-1][0], obj['test_metric'][-1][1]]),
					'count':1
				}
			else:
				res[param]['data']+=np.array([obj['test_metric'][-1][0], obj['test_metric_attack'][-1][0], obj['test_metric'][-1][1]])
				res[param]['count']+=1
		f.close()

	for param in res:
		res[param]=res[param]['data']/res[param]['count']

	ret = []

	for wF in wF_list:
		for wR in wR_list:
			item = res[(wR, wF)]
			if wF == 0.0:
				ret.append([item[0], item[1]])
			print('(%.4f, %.4f, %.4f)'%(item[0], item[1], item[2]), end='')
			if wR != wR_list[-1]:
				print('\t',end='')
		print()
			# else:
			# 	print(' \\\\\n\\hline')
	print(ret)
	return np.array(ret)

res = accuracy('compas','race')#, 1.0, 1.0)

from matplotlib import pyplot as plt
plt.scatter(res[:,0], res[:,1], label='InProc', alpha=0.5)

import json
f=open('../../preproc/Robustness.txt')
z = json.loads(f.readline())
f.close()

res=[]
for i in range(0,len(z['test']['orig'])):
	res.append([z['test']['orig'][i], z['test']['attk'][i]])
res = np.array(res)

plt.scatter(res[:,0], res[:,1], label='PreProc', alpha=0.5)
plt.legend()
plt.xlabel('Accuracy')
plt.ylabel('Attacked_Accuracy')
plt.title(f"{z['data']}_{z['attr']}")

plt.show()







