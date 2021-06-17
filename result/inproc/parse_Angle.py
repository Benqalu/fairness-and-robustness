import numpy as np
import gzip, json

def angle(data, attr, method, wR, wF):
	epochs = None
	res = {}
	for i in range(1,12):
		f=gzip.open('./RnF_%d.txt.gz'%(i),'rt')
		for row in f:
			obj=json.loads(row)
			if epochs is None:
				epochs=obj['epoch']
			if (obj['data'], obj['attr'], obj['method']) != (data, attr, method):
				continue
			param = (obj['wR'], obj['wF'])
			if param!=(wR, wF):
				continue
			if len(obj['angle_rf'])<10:
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

angle('adult', 'race', 'FGSM', 0.1, 0.4)







