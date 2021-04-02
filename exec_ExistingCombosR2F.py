import os, time, json

count=0
allct=10*2*2*3
runtime=0.0

executed={}

if os.path.exists('./result/existings/R2F.txt'):
	f=open('./result/existings/R2F.txt')
	for line in f:
		obj=json.loads(line)
		combo = (obj['data'], obj['attr'], obj['method'])
		if combo not in executed:
			executed[combo]=0
		executed[combo]+=1
		allct-=1

for it in range(1,11):
	try:
		for data in ['compas','adult']:
			for attr in ['race','sex']:
				for method in ['FGSM','PGD','None']:
					combo = (data, attr, method)
					if combo in executed and executed[combo]>0:
						print('Combo %s skpped.'%str(combo))
						executed[combo]-=1
						continue

					start_t=time.time()

					os.system('python ExistingCombosR2F.py %s %s %s'%(data, attr, method))
					count+=1
					time.sleep(1)

					end_t=time.time()

					runtime+=end_t-start_t

					print('Finished %d / %d'%(count, allct))
					print('Runtime: %.2f seconds, Avg.: %.2f, Est.: %.2f'%(end_t-start_t, runtime/count, (allct-count)*runtime/count))
					print('')
						
						

	except KeyboardInterrupt:
		break