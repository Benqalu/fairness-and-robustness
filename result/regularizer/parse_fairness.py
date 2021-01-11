import numpy as np
from matplotlib import pyplot as plt

def draw(ax, data, attr, metric='acc'):
	f=open('./epoch_fairness_%s_%s_24.txt'%(data,attr),'r')
	l1=np.array(eval(f.readline()))
	l2=np.array(eval(f.readline()))

	cmap = plt.get_cmap("tab10")

	x=l1[:,0]
	if metric=='acc':
		y=[l1[:,1], l1[:,2], l2[:,1], l2[:,2]]
	if metric=='disp':
		y=[l1[:,3], l1[:,4], l2[:,3], l2[:,4]]
		ylim_max=np.max(y)
		ax.set_ylim(0.0, ylim_max*2)

	ax.plot(x,y[0],label='w/o_F_train', color=cmap(0))
	ax.plot(x,y[1],label='w/o_F_test', linestyle='--', color=cmap(0))
	ax.plot(x,y[2],label='w/__F_train', color=cmap(1))
	ax.plot(x,y[3],label='w/__F_test', linestyle='--', color=cmap(1))

	ax.set_xlabel('Number of epochs')
	if metric=='acc':
		ax.set_ylabel('Accuracy')
	if metric=='disp':
		ax.set_ylabel('Positive disparity')
	ax.set_title('%s_%s'%(data,attr))
	ax.legend()

fig, axs = plt.subplots(3,2)

datas=['adult','compas','hospital']
attrs=['race','sex']

for data in datas:
	plt.clf()
	fig, axs = plt.subplots(2,2)
	for j in range(0,2):
		draw(ax=axs[j][0], data=data, attr=attrs[j], metric='acc')
		draw(ax=axs[j][1], data=data, attr=attrs[j], metric='disp')
	plt.tight_layout()
	plt.savefig('fairness_%s.pdf'%data)

