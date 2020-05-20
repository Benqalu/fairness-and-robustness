import os
import numpy as np
from matplotlib import pyplot as plt

def parse(fname):

	print(fname)

	f=open(fname)
	orig=eval(f.readline())
	fair=eval(f.readline())
	f.close()

	for i in range(0,len(orig['original'])):
		orig['original'][i]=orig['original'][i][1]
		orig['adversial'][i]=orig['adversial'][i][1]
		fair['original'][i]=fair['original'][i][1]
		fair['adversial'][i]=fair['adversial'][i][1]

	print(abs((np.array(orig['original'])-0.5)).mean(),abs((np.array(fair['original'])-0.5)).mean())

	bins=[i*0.05 for i in range(0,21)]

	s=fname.split('_')[2]
	z=fname.split('_')[3]

	plt.clf()
	plt.subplot(221)
	plt.hist(orig['original'],bins=bins,edgecolor='black')
	plt.title(s+'_'+z+'_original')
	plt.subplot(222)
	plt.hist(orig['adversial'],bins=bins,edgecolor='black')
	plt.title(s+'_'+z+'_original_attack')
	plt.subplot(223)
	plt.hist(fair['original'],bins=bins,edgecolor='black')
	plt.title(s+'_'+z+'_adjusted')
	plt.subplot(224)
	plt.hist(fair['adversial'],bins=bins,edgecolor='black')
	plt.title(s+'_'+z+'_adjusted_attack')
	plt.tight_layout()
	# plt.show()
	plt.savefig(s+'_'+z+'_distribution.png')

if __name__=='__main__':
	fnames=os.listdir('.')
	for fname in fnames:
		if '.txt' not in fname:
			continue
		parse(fname)