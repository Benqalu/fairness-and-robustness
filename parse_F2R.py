import json
import numpy as np

data = {}

f=open('./result/existings/F2R.txt')
for line in f:
	obj=json.loads(line)
	combo=(obj['method'],obj['func'],obj['data'],obj['attr'],)
	if combo not in data:
		data[combo]={
			'orig':np.array([0.0,0.0]),
			'fair':np.array([0.0,0.0]),
			'count':0
		}
	data[combo]['orig']+=np.array([obj['result_orig']['test_acc'],obj['result_orig']['test_adv_acc']])
	data[combo]['fair']+=np.array([obj['result_fair']['test_acc'],obj['result_fair']['test_adv_acc']])
	data[combo]['count']+=1
f.close()

for combo in data:
	data[combo]['orig']/=data[combo]['count']
	data[combo]['fair']/=data[combo]['count']
	data[combo]['score_orig']=1-data[combo]['orig'][1]
	data[combo]['score_fair']=1-data[combo]['fair'][1]
	data[combo]['score']=(data[combo]['score_orig']-data[combo]['score_fair'])/data[combo]['score_orig']

i=0
for combo in sorted(data.keys()):
	print(combo, data[combo])
	# print(round(data[combo]['score'],4),end=' & ')
	# i+=1
	# if i%4==0:
	# 	print()
