import os, time, json

count=0
allct=30*2*2*2*2
runtime=0.0

executed={}

if os.path.exists('./result/existings/R2F.txt'):
	f=open('./result/existings/R2F.txt')
	for line in f:
		obj=json.loads(line)
		combo = (obj['data'], obj['attr'], obj['method'], obj['defense'])
		if combo not in executed:
			executed[combo]=0
		executed[combo]+=1
		allct-=1

for it in range(1,31):
	try:
		for data in ['compas','adult']:
			for attr in ['race','sex']:
				for method in ['FGSM','PGD']:
					for defense in [0.0,1.0]:

						combo = (data, attr, method, defense)
						if combo in executed and executed[combo]>0:
							executed[combo]-=1
							continue

						start_t=time.time()

						os.system('python ExistingCombosR2F.py %s %s %s %s'%(data, attr, method, defense))
						count+=1
						time.sleep(1)

						end_t=time.time()

						runtime+=end_t-start_t

						print('Finished %d / %d'%(count, allct))
						print('Runtime: %.2f seconds, Avg.: %.2f, Est.: %.2f'%(end_t-start_t, runtime/count, (allct-count)*runtime/count))
						print('')
						
						

	except KeyboardInterrupt:
		break