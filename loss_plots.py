import os
import numpy as np
from matplotlib import pyplot as plt

alphas=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
betas=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

datas=['adult','compas']
attrs=['sex','race']

path='./result/lr_positive_disp'

fnames=os.listdir(path)

xx=[i*10 for i in range(0,300)]

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

content={}
for fname in fnames:
	if '.txt' not in fname:
		continue
	components=fname.split('_')
	data=components[1]
	attr=components[2]
	alpha=int(components[3][1:])/100
	beta=int(components[4][1:])/100
	if data not in content:
		content[data]={}
	if attr not in content[data]:
		content[data][attr]={}
	if alpha not in content[data][attr]:
		content[data][attr][alpha]={}
	if beta not in content[data][attr][alpha]:
		content[data][attr][alpha][beta]={}
	f=open(path+'/'+fname)
	z=eval(f.readline())
	f.close()
	content[data][attr][alpha][beta]['loss_utility']=np.array(z['loss_utility'])
	content[data][attr][alpha][beta]['loss_fairness']=np.array(z['loss_fairness'])
	content[data][attr][alpha][beta]['loss_robustness']=np.array(z['loss_robustness'])

	# content[data][attr][alpha][beta]['loss_utility']=np.log(content[data][attr][alpha][beta]['loss_utility']+1)
	# content[data][attr][alpha][beta]['loss_fairness']=np.log(content[data][attr][alpha][beta]['loss_fairness']+1)
	# content[data][attr][alpha][beta]['loss_robustness']=np.log(content[data][attr][alpha][beta]['loss_robustness']+1)

	fig, axs = plt.subplots()
	# axs.plot(xx, content[data][attr][alpha][beta]['loss_utility'], label='loss_utility')
	color=cycle[0]
	axs.plot(xx, content[data][attr][alpha][beta]['loss_fairness'], label='loss_fairness', color=color)
	axs.set_xlabel('n_epochs')
	axs.set_ylabel('loss_fairness',color=color)
	axs.tick_params(axis='y', labelcolor=color)
	axs.legend()

	color=cycle[1]
	ax2=axs.twinx()
	ax2.plot(xx, content[data][attr][alpha][beta]['loss_robustness'], label='loss_robustness',color=color)
	ax2.set_ylabel('loss_robustness',color=color)
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.set_title('%s_%s_f%s_r%s'%(data,attr,str(alpha),str(beta)))
	ax2.legend()

	name=fname.split('_')
	name[0]='loss'
	name='_'.join(name)
	name=name[:-3]+'pdf'

	fig.tight_layout()
	plt.plot()
	plt.savefig('./result/lr_positive_disp/loss/'+name)

# for data in datas:
# 	for attr in attrs:
# 		plt.clf()

# 		fig, axs = plt.subplots(len(alphas), len(betas))
# 		for i in range(0, len(alphas)):
# 			for j in range(0, len(betas)):
# 				axs[i][j].plot(xx, content[data][attr][alphas[i]][betas[j]]['loss_utility'], label='loss_utility')
# 				axs[i][j].plot(xx, content[data][attr][alphas[i]][betas[j]]['loss_fairness'], label='loss_fairness')
# 				axs[i][j].plot(xx, content[data][attr][alphas[i]][betas[j]]['loss_robustness'], label='loss_robustness')
# 		fig.tight_layout()
# 		plt.plot()
# 		plt.show()
# 		exit()



