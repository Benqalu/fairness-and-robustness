import numpy as np
from matplotlib import pyplot as plt

f=open('adult_sex.txt','r')
lines=f.readlines()
f.close()

fig,axs=plt.subplots(4)
fig.set_size_inches(16, 9)

data={}
for line in lines:
	line=eval(line)
	if line['beta']==0:
		data[line['alpha']]=line['origin_pred']

hists=[]
alphas=sorted(data.keys())
labels=[f'alpha={alpha}' for alpha in alphas]
for alpha in alphas:
	hists.append(data[alpha])
axs[0].hist(hists,label=labels, bins=[i*0.05 for i in range(0,21)], alpha=0.9)
axs[0].set_title('adult_sex, fairness_vs_distribution, origin')
axs[0].set_xlabel('Predicted probability')
axs[0].legend(ncol=3)

data={}
for line in lines:
	line=eval(line)
	if line['beta']==0:
		data[line['alpha']]=line['attack_pred']

hists=[]
alphas=sorted(data.keys())
labels=[f'alpha={alpha}' for alpha in alphas]
for alpha in alphas:
	hists.append(data[alpha])
axs[1].hist(hists,label=labels, bins=[i*0.05 for i in range(0,21)], alpha=0.9)
axs[1].set_title('adult_sex, fairness_vs_distribution, attack')
axs[1].set_xlabel('Predicted probability')
axs[1].legend()
axs[1].legend(ncol=3)


data={}
for line in lines:
	line=eval(line)
	if line['alpha']==0:
		data[line['beta']]=line['origin_pred']
hists=[]
betas=sorted(data.keys())
labels=[f'beta={beta}' for beta in betas]
for beta in betas:
	hists.append(data[beta])
axs[2].hist(hists,label=labels, bins=[i*0.05 for i in range(0,21)], alpha=0.9)
axs[2].set_title('adult_sex, robustness_vs_distribution, origin')
axs[2].set_xlabel('Predicted probability')
axs[2].legend(ncol=3)

data={}
for line in lines:
	line=eval(line)
	if line['alpha']==0:
		data[line['beta']]=line['attack_pred']
hists=[]
betas=sorted(data.keys())
labels=[f'beta={beta}' for beta in betas]
for beta in betas:
	hists.append(data[beta])
axs[3].hist(hists,label=labels, bins=[i*0.05 for i in range(0,21)], alpha=0.9)
axs[3].set_title('adult_sex, robustness_vs_distribution, attack')
axs[3].set_xlabel('Predicted probability')
axs[3].legend(ncol=3)


fig.tight_layout()
plt.savefig('distribution_adult_sex.pdf')
plt.show()

