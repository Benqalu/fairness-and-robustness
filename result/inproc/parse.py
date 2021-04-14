import gzip, json
import numpy as np
from matplotlib import pyplot as plt

class InProcessDrawer(object):
	def __init__(self, method='FGSM'):
		data = {}
		for i in range(1,12):
			f=gzip.open('./%s/RnF_%d.txt.gz'%(method, i),'rt')
			for line in f:
				obj = json.loads(line)
				print(obj['wR'], obj['wF'])
				print(obj['angle_rf'])
				combo = (obj['data'], obj['attr'], obj['wR'], obj['wF'])
				if combo not in data:
					data[combo]={}
					data[combo]['test_origin']=np.array([0.,0.])
					data[combo]['test_attack']=np.array([0.,0.])
					data[combo]['count']=0
				data[combo]['test_origin']+=np.array(obj['test_metric'][-1])
				data[combo]['test_attack']+=np.array(obj['test_metric_attack'][-1])
				data[combo]['count']+=1
			f.close()
		for combo in data:
			data[combo]['test_origin'] /= data[combo]['count']
			data[combo]['test_attack'] /= data[combo]['count']
			del data[combo]['count']
		self.data = data
		self.x = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281, 286, 291, 296, 301, 306, 311, 316, 321, 326, 331, 336, 341, 346, 351, 356, 361, 366, 371, 376, 381, 386, 391, 396, 401, 406, 411, 416, 421, 426, 431, 436, 441, 446, 451, 456, 461, 466, 471, 476, 481, 486, 491, 496, 499]

	def accuracy(self, data, attr, wF = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], y_axis='Disparity'):
		# x=[0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
		plt.clf()
		for wF_ in wF:
			x=[]
			y=[]
			for item in sorted(self.data.keys()):
				if (item[0], item[1], item[3]) != (data, attr, wF_):
					continue
				x.append(item[2])
				if y_axis=='Disparity':
					y.append(self.data[item]['test_origin'][1])
				elif y_axis=='Accuracy':
					y.append(self.data[item]['test_origin'][0])
				elif y_axis=='Accuracy_Attack':
					y.append(self.data[item]['test_attack'][0])
			plt.plot(x,y[:len(x)],label='wF=%.2f'%wF_)
			print('wF=%.1f'%wF_, end='')
			for i in range(0,len(x)):
				print((x[i], y[i]), end=' ')
			print()

		plt.xlabel('wR')
		plt.ylabel(y_axis)
		plt.legend()
		plt.savefig(f'{data}_{attr}_{y_axis}.pdf')
		# plt.show()


if __name__=='__main__':
	z = InProcessDrawer()
	for data in ['adult','compas']:
		for attr in ['race', 'sex']:
			for y_axis in ['Disparity', 'Accuracy', 'Accuracy_Attack']:
				z.accuracy(data=data, attr=attr, y_axis=y_axis)

