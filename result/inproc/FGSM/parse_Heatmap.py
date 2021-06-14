import matplotlib, gzip, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
import seaborn as sns

font = {'size': 18}

matplotlib.rc('font', **font)

def draw(data, attr, method):
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
	Fs = []
	Rs = []
	As = []
	for param in res:
		res[param]=res[param]['data']/res[param]['count']
		As.append(res[param][0])
		Rs.append(res[param][1])
		Fs.append(res[param][2])

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
	print(rr)
	aa = interpolate.griddata((Rs, Fs), As, (rr.ravel(), ff.ravel()))
	# xx.ravel(), yy.ravel()
	
	dataset = pd.DataFrame(data={'x':rr.ravel(), 'y':ff.ravel(), 'z':aa})
	dataset = dataset.pivot(index='y', columns='x', values='z')
	print(dataset.to_numpy().shape)
	# heatmap=plt.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy())
	# plt.colorbar(heatmap)

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy())
	cbar = plt.colorbar(heatmap)
	ax.set_xticks(np.arange(np.ceil(np.min(Rs)*10)/10,np.floor(np.max(Rs)*10)/10+0.1,0.1))

	plt.xlabel('Robustness')
	plt.ylabel('Fairness')
	plt.tight_layout()
	plt.savefig(f'heatmap_{data}_{attr}_{method}.pdf')
	
	res={
		'data':data,
		'attr':attr,
		'method':'FGSM',
		'xticks':np.arange(np.ceil(np.min(Rs)*10)/10,np.floor(np.max(Rs)*10)/10+0.1,0.1).tolist(),
		'R':rr_.ravel().tolist(),
		'F':ff_.ravel().tolist(),
		'A':dataset.to_numpy().tolist()
	}

	f=open(f'heatmap_result_{data}_{attr}_{method}.txt','w')
	f.write(json.dumps(res))
	f.close()
	

if __name__=='__main__':
	for data in ['adult','compas','hospital']:
		for attr in ['race', 'sex']:
			for method in ['FGSM', 'PGD']:
				try:
					draw(data, attr, method)
				except:
					pass