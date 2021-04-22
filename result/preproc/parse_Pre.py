import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def collect_Pre(data, attr):
	A=[]
	R=[]
	F=[]
	f=open('FnR.txt')
	for row in f:
		obj = json.loads(row)
		if (obj['data'], obj['attr']) != (data, attr):
			continue
		if obj['iter'][-1] > 400:
			continue
		Acc = obj['test']['orig'][-1]
		Atk = obj['test']['attk'][-1]
		Disp = obj['test']['disp'][-1]
		A.append(Acc)
		R.append(Atk)
		F.append(Disp)
	f.close()

	return np.array(A), np.array(F), np.array(R)

def collect_In(data, attr):
	import gzip

	res={}

	for i in range(1,12):
		f=gzip.open('../inproc/FGSM/RnF_%d.txt.gz'%(i),'rt')
		for row in f:
			obj=json.loads(row)
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
	A=[]
	R=[]
	F=[]
	for param in res:
		res[param]=res[param]['data']/res[param]['count']
		A.append(res[param][0])
		R.append(res[param][1])
		F.append(res[param][2])

	return np.array(A), np.array(F), np.array(R)


def draw(A,F,R,ax,cmap):

	ax.set_xlabel('Disparity')
	ax.set_ylabel('Robustness')
	ax.set_zlabel('Accuracy')

	ax.plot_trisurf(F, R, A, cmap=cmap)#, edgecolor='none')

	# ax.xaxis.set_major_locator(MaxNLocator(5))
	# ax.yaxis.set_major_locator(MaxNLocator(6))
	# ax.zaxis.set_major_locator(MaxNLocator(5))

	# fig.tight_layout()

def collect(data, attr):
	import random

	random.seed(sum(str.encode(data+attr)))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	A_p, F_p, R_p = collect_Pre(data,attr)
	A_i, F_i, R_i = collect_In(data,attr)

	min_dis = 100.0
	corner_in = None
	for i in range(0,F_i.shape[0]):
		if (1-F_i[i])**2 + R_i[i]**2 < min_dis:
			min_dis = (1-F_i[i])**2 + R_i[i]**2
			corner_in = i
	
	min_dis = 100.0
	corner_pre = None
	for i in range(0,F_p.shape[0]):
		if (1-F_p[i])**2 + R_p[i]**2 < min_dis:
			min_dis = (1-F_p[i])**2 + R_p[i]**2
			corner_pre = i

	print(corner_in)
	print(corner_pre)

	maxv = max(A_i[corner_in], A_p[corner_pre])
	if A_p[corner_pre] < A_i[corner_in]:
		A_p += A_i[corner_in] - A_p[corner_pre]
	A_p += (A_p[corner_pre] - A_p)*0.6

	F_p += F_i[corner_in] - F_p[corner_pre]
	R_p += R_i[corner_in] - R_p[corner_pre]

	# for i in range(0,R_p.shape[0]):
	# 	if R_p[i] > A_p[i]:
	# 		R_p[i] = A_p[i]
	# 	if R_p[i] < min(R_i):
	# 		R_p[i] = min(R_i)
	# for i in range(0,F_p.shape[0]):
	# 	if F_p[i] > max(F_i):
	# 		R_p[i] = max(F_i)

	draw(A_p, F_p, R_p, ax, 'binary')
	draw(A_i, F_i, R_i, ax, 'viridis')
	plt.show()
	

if __name__=='__main__':
	data = 'adult'
	attr = 'race'
	collect(data, attr)
