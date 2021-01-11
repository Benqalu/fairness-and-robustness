import numpy as np
from matplotlib import pyplot as plt

def draw(ax, data, attr='race', method='offset', ylim=None):
	f=open('./robustness_%s_%s_%s_24.txt'%(data,attr,method),'r')
	line=np.array(eval(f.readline()))

	ax.plot(line[:,0],line[:,1],label='Orig_Acc')
	ax.plot(line[:,0],line[:,2],label='Attk_Acc')
	ax.legend()
	ax.set_xlabel('Regularizer factor')
	ax.set_ylabel('Accuracy')

	names={'offset':'offset','switch':'flip'}

	ax.set_title('%s_%s'%(data,names[method]))

	return ax.get_ylim()

fig, axs = plt.subplots(3,2)

datas=['adult','compas','hospital']
attrs=['race','sex']

for data in datas:
	plt.clf()
	fig, axs = plt.subplots(1,2)
	fig.set_size_inches(8, 3)
	draw(ax=axs[0], data=data, method='offset')
	draw(ax=axs[1], data=data, method='switch')
	plt.tight_layout()
	plt.savefig('robustness_%s.pdf'%data)

