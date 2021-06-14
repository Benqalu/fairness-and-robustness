import os

config = {
	('adult', 'race', 'FGSM'): [
		[0.01, 0.7],
		[0.03, 0.6],
		[0.05, 0.5],
	],
	('adult', 'sex', 'FGSM'): [
		[0.01, 0.7],
		[0.05, 0.6],
		[0.10, 0.5],
	],
	('compas', 'race', 'FGSM'): [
		[0.01, 0.65],
		[0.05, 0.55],
		[0.10, 0.45],
	],
	('compas', 'sex', 'FGSM'): [
		[0.01, 0.65],
		[0.015, 0.55],
		[0.02, 0.45],
	],
	('hospital', 'race', 'FGSM'): [
		[0.01, 0.55],
		[0.015, 0.45],
		[0.02, 0.35],
	],
	('hospital', 'sex', 'FGSM'): [
		[0.01, 0.55],
		[0.015, 0.45],
		[0.02, 0.35],
	],
	('adult', 'race', 'PGD'): [
		[0.01, 0.7],
		[0.03, 0.6],
		[0.05, 0.5],
	],
	('adult', 'sex', 'PGD'): [
		[0.01, 0.7],
		[0.05, 0.6],
		[0.10, 0.5],
	],
	('compas', 'race', 'PGD'): [
		[0.01, 0.65],
		[0.05, 0.55],
		[0.10, 0.45],
	],
	('compas', 'sex', 'PGD'): [
		[0.01, 0.65],
		[0.015, 0.55],
		[0.02, 0.45],
	],
	('hospital', 'race', 'PGD'): [
		[0.01, 0.55],
		[0.015, 0.45],
		[0.02, 0.35],
	],
	('hospital', 'sex', 'PGD'): [
		[0.01, 0.55],
		[0.015, 0.45],
		[0.02, 0.35],
	],
}

def Generate(n=20):

	# executed = {}
	# for item in config:
	# 	if item not in executed:
	# 		executed[item] = n

	# fnames = os.listdir('./result/predata/')
	# for fname in fnames:
	# 	item = fname.split('.')[0].split('_')
	# 	data = item[0]
	# 	attr = item[1]
	# 	method = item[2]
	# 	executed[(data,attr,method)] -= 1

	for t in range(0,n):
		for data, attr, method in config:
			if data != 'hospital':
				continue
			for item in config[(data, attr, method)]:
				wF, wR = item[0], item[1]
				os.system(
					"python PreProcessGen.py %s %s %s %.3f %.3f"
					% (data, attr, method, wF, wR)
				)

def Downstreams():
	import pickle
	from PreProcessGen import PreProcFlip

	fnames = []
	for fname in sorted(os.listdir('./result/predata/')):
		if '.pre' not in fname:
			continue
		item = fname.split('.')[0].split('_')
		data = item[0]
		if data != 'hospital':
			continue
		attr = item[1]
		method = item[2]
		seed = int(item[3])
		fnames.append((seed, data, attr, method))
	fnames.sort()

	for fname in fnames:
		model = PreProcFlip(fname[1], fname[2], k=0.003, method=fname[3], max_iter=1000, seed=seed, dR=0.0, dF=0.0)
		fname = f'{fname[1]}_{fname[2]}_{fname[3]}_{fname[0]}.pre'
		res = model.downstreams(loadfrom=fname)
		print(fname)
		for item in res:
			for jtem in item:
				print(round(jtem,4),end='\t')
			print()
		print()

# Generate(n=20)
Downstreams()






