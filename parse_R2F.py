import json
import numpy as np

data = {}

f=open('./result/existings/R2F.txt')
for line in f:
	obj=json.loads(line)
	combo=(obj['data'],obj['attr'])
	if combo not in data:
		data[combo]={'count':0,}
	if obj['method'] not in data[combo]:
		data[combo][obj['method']]=np.array([0.,0.])
	data[combo][obj['method']]+=np.array(obj['test'])
f.close()

for combo in data:
	for item in data[combo]:
		data[combo][item]/=10

print(data)

for method in ['FGSM', 'PGD']:
	for combo in sorted(data.keys()):
		change = (data[combo][method]-data[combo]['None'])/data[combo]['None']
		print(combo, method, round(change[1],4), data[combo][method])