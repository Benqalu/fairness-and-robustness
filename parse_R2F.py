import json
import numpy as np

data = {}

f=open('./result/existings/R2F.txt')
for line in f:
	obj=json.loads(line)
	combo=(obj['method'],obj['defense'],obj['data'],obj['attr'])
	if combo not in data:
		data[combo]={
			'disps':np.array([0.0,0.0]),
			'count':0,
		}
	data[combo]['disps']+=np.array([obj['test']['disp'],obj['test_'+obj['method'].lower()]['disp']])
	data[combo]['count']+=1
f.close()

for combo in data:
	data[combo]['disps']/=data[combo]['count']
	data[combo]['score']=(data[combo]['disps'][1]-data[combo]['disps'][0])/data[combo]['disps'][0]

i=0
for combo in sorted(data.keys()):
	if combo[1]==0.0:
		continue
	print(combo, data[combo])
	# print(round(data[combo]['score'],4), end=' & ')
	# i+=1
	# if i%4==0:
	# 	print()