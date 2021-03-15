import os
from time import sleep

prngs={
	'fairness_reweighing':1,
	'fairness_disparate':21,
	'fairness_adversarial':21,
}

try:
	for _ in range(100):
		for func in ['fairness_reweighing', 'fairness_disparate', 'fairness_adversarial']:
			for method in ['FGSM','PGD']:
				for data in ['compas','adult']:
					for attr in ['race','sex']:
						for pidx in range(prngs[func]):
							# print(func, method, data, attr, pidx)
							os.system('python ExistingCombos.py %s %s %s %s %s'%(func, method, data, attr, pidx))
							sleep(3)
except KeyboardInterrupt:
	pass