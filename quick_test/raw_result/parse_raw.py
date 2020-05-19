import os
from matplotlib import pyplot as plt

def parse(fname):

	print(fname)

	f=open(fname)
	orig=eval(f.readline())
	fair=eval(f.readline())
	f.close()

	orig=orig['original']
	fair=fair['original']

	for i in range(0,len(orig)):
		orig[i]=orig[i][1]
		fair[i]=fair[i][1]
	
	plt.hist(orig)
	plot.show()

if __name__=='__main__':
	fnames=os.listdir('.')
	for fname in fnames:
		if '.txt' not in fname:
			continue
		parse(fname)