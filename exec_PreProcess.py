import os, time, json

turn=10

count=0
runtime=0.0
allct=0

dFs = [round(0.01*i,2) for i in range(1,21)]
dRs = [round(0.05*i,2) for i in range(0,20)]

executed={}

for it in range(1,turn+1):
	for data in ['compas','adult']:
		for attr in ['race','sex']:
			for method in ['FGSM']:#,'PGD']:
				for dF in dFs:
					for dR in dRs:
						combo = (data, attr, method, func, wF, wR)
						if combo not in executed:
							executed[combo]=0
						allct+=1

if os.path.exists('./result/preproc/FnR_Pre.txt'):
	f=open('./result/preproc/FnR_Pre.txt')
	for line in f:
		obj=json.loads(line)
		combo = (obj['data'], obj['attr'], obj['method'], obj['dF'], obj['dR'])
		if combo not in executed:
			executed[combo]=0
		executed[combo]+=1
		allct-=1

for it in range(1,turn+1):
	try:
		for data in ['compas','adult']:
			for attr in ['race','sex']:
				for method in ['FGSM']:#,'PGD']:
					for dF in dFs:
						for dR in dRs:

							combo = (data, attr, method, dF, dR)

							if combo in executed and executed[combo]>0:
								executed[combo]-=1
								print('Combo %s is already calculated, skip.'%str(combo))
								continue

							start_t=time.time()

							os.system('python PreProcess.py %s %s %s %.2f %.2f'%(data, attr, method, dF, dR))
							count+=1
							time.sleep(1)

							end_t=time.time()

							runtime+=end_t-start_t

							print('Finished %d / %d'%(count, allct))
							print('Runtime: %.2f seconds, Avg.: %.2f, Est.: %.2f'%(end_t-start_t, runtime/count, (allct-count)*runtime/count))
							print('')
						
	except KeyboardInterrupt:
		break