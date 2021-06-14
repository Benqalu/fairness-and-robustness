import os, time, json

count=0
allct=10*2*2*2*(1+5)
runtime=0.0

executed={}

if os.path.exists('./result/existings/F2R.txt'):
	f=open('./result/existings/F2R.txt')
	for line in f:
		obj=json.loads(line)
		combo = (obj['data'], obj['attr'], obj['method'], obj['func'], obj['pidx'])
		if combo not in executed:
			executed[combo]=0
		executed[combo]+=1
		allct-=1

param_range={
	'fairness_reweighing':1,
	'fairness_disparate':5,
}

for it in range(1,11):
	try:
		for data in ['compas','adult']:
			for attr in ['race','sex']:
				for method in ['FGSM','PGD']:
					for func in ['fairness_reweighing','fairness_disparate']:#,'fairness_adversarial']:
						for pidx in range(param_range[func]):
							combo = (data, attr, method, func, pidx)
							if combo in executed and executed[combo]>0:
								executed[combo]-=1
								print('Combo %s is already calculated, skip.'%str(combo))
								continue

							start_t=time.time()

							os.system('python ExistingCombosF2R.py %s %s %s %s %d'%(data, attr, method, func, pidx))
							count+=1
							time.sleep(1)

							end_t=time.time()

							runtime+=end_t-start_t

							print('Finished %d / %d'%(count, allct))
							print('Runtime: %.2f seconds, Avg.: %.2f, Est.: %.2f'%(end_t-start_t, runtime/count, (allct-count)*runtime/count))
							print('')
						
	except KeyboardInterrupt:
		break