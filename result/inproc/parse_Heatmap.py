import matplotlib, gzip, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
import seaborn as sns

font = {'size': 22 }
matplotlib.rc('font', **font)

minmax={}

def collect_colorbar_range(data):
	if data in minmax:
		return minmax[data][0], minmax[data][1]
	Accs = []
	for i in range(1,12):
		f=gzip.open('./RnF_%d.txt.gz'%(i),'rt')
		for row in f:
			obj=json.loads(row)
			if obj['data'] != data:
				continue
			Accs.append(obj['test_metric'][-1][0])
	minmax[data] = (min(Accs), max(Accs))
	return min(Accs), max(Accs)

def draw(data, attr, method):

	print(data, attr, method)

	vmin = None
	vmax = None
	# vmin, vmax = collect_colorbar_range(data)

	plt.clf()
	res = {}
	for i in range(1,12):
		f=gzip.open('./RnF_%d.txt.gz'%(i),'rt')
		for row in f:
			obj=json.loads(row)
			if (obj['data'], obj['attr'], obj['method']) != (data, attr, method):
				continue
			wR = obj['wR']
			wF = obj['wF']
			param = (wR, wF)
			if param not in res:
				res[param]={
					'data':np.zeros(3),
					'count':0
				}
			res[param]['data']+=np.array([obj['test_metric'][-1][0], obj['test_metric_attack'][-1][0], obj['test_metric'][-1][1]])
			res[param]['count']+=1
		f.close()

	plain_acc = 0.0
	worst_acc = 1.0
	wrost_case = None

	Fs = []
	Rs = []
	As = []
	for param in res:
		res[param]=res[param]['data']/res[param]['count']
		As.append(res[param][0])
		Rs.append(res[param][1])
		Fs.append(res[param][2])

		if param==(0,0):
			plain_acc=res[param][0]
		if res[param][0]<worst_acc:
			worst_acc=res[param][0]
			wrost_case = tuple(res[param].tolist())
	print(plain_acc, worst_acc, plain_acc-worst_acc)
	print(wrost_case)
	exit()

	mapdata = {
		'A':As,
		'R':Rs,
		'F':Fs,
	}
	f=open(f'../preindiff/heatmap_inproc_{data}_{attr}_{method}.txt','w')
	f.write(json.dumps(mapdata))
	f.close()

	# ff = np.linspace(np.min(Fs), np.max(Fs))
	# rr = np.linspace(np.min(Rs), np.max(Rs))
	fmin=np.floor(np.min(Fs)*100)/100
	fmax=np.ceil(np.max(Fs)*100)/100
	# rmin=np.floor(np.min(Rs)*10)/10
	# rmax=np.ceil(np.max(Rs)*10)/10
	print(fmin, fmax)
	print(np.min(Rs), np.max(Rs))
	print(np.min(Fs), np.max(Fs))
	ff_ = np.linspace(np.min(Fs), np.max(Fs) ,num=101)
	rr_ = np.linspace(np.min(Rs),np.max(Rs),num=101)
	rr, ff = np.meshgrid(rr_, ff_)
	aa = interpolate.griddata((Rs, Fs), As, (rr.ravel(), ff.ravel()))
	# xx.ravel(), yy.ravel()
	
	dataset = pd.DataFrame(data={'x':rr.ravel(), 'y':ff.ravel(), 'z':aa})
	dataset = dataset.pivot(index='y', columns='x', values='z')
	print(dataset.to_numpy().shape)
	# heatmap=plt.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy())
	# plt.colorbar(heatmap)

	fig, ax = plt.subplots()
	if vmin==vmax==None:
		heatmap = ax.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy())
	else:
		heatmap = ax.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy(), vmin=vmin, vmax=vmax)
	cbar = plt.colorbar(heatmap)
	cbar.set_label('Accuracy')

	xtick_min = np.floor(np.min(Rs)*100)/100
	xtick_max = np.ceil(np.max(Rs)*100)/100
	xtick_bins = 5
	xtick_step = (xtick_max - xtick_min) / (xtick_bins - 1)
	xticks = np.round(np.arange(xtick_min, xtick_max, xtick_step), 2).tolist()
	if xtick_max - xticks[-1] > 0.75*xtick_step:
		xticks.append(xtick_max)
	print(xticks)
	ax.set_xticks(xticks)

	# ytick_min = np.floor(np.min(Fs)*100)/100
	# ytick_max = np.ceil(np.max(Fs)*100)/100
	# ytick_bins = 5
	# ytick_step = (ytick_max - ytick_min) / (ytick_bins - 1)
	# yticks = np.round(np.arange(ytick_min, ytick_max, ytick_step), 3).tolist()
	# if ytick_max - yticks[-1] > 0.75*ytick_step:
	# 	yticks.append(ytick_max)
	# print(yticks)
	# ax.set_yticks(yticks)

	ztick_min = np.ceil(np.min(As)*1000)/1000
	ztick_max = np.floor(np.max(As)*1000)/1000
	ztick_bins = 5
	ztick_step = (ztick_max - ztick_min) / (ztick_bins - 1)
	zticks = np.round(np.arange(ztick_min, ztick_max, ztick_step), 3).tolist()
	if ztick_max - zticks[-1] > 0.75*ztick_step:
		zticks.append(ztick_max)
	print(zticks)
	cbar.set_ticks(zticks)

	# ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='y')

	plt.xlabel('Robustness score')
	plt.ylabel('Bias score')
	plt.tight_layout()
	plt.savefig(f'heatmap_inproc_{data}_{attr}_{method}.pdf')
	
	res={
		'data':data,
		'attr':attr,
		'method':'FGSM',
		'xticks':np.arange(np.ceil(np.min(Rs)*10)/10,np.floor(np.max(Rs)*10)/10+0.1,0.1).tolist(),
		'R':rr_.ravel().tolist(),
		'F':ff_.ravel().tolist(),
		'A':dataset.to_numpy().tolist()
	}
	

if __name__=='__main__':
	for data in ['adult','compas','hospital']:
		for attr in ['race', 'sex']:
			for method in ['FGSM', 'PGD']:
				try:
					draw(data, attr, method)
				except:
					pass