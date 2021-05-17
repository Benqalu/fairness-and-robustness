import json

data=[]
f=open('FnR.txt')
for row in f:
	row=json.loads(row)
	row['method']='FGSM'
	data.append(json.dumps(row))
f.close()

f=open('FnR_New.txt','w')
for item in data:
	f.write(item+'\n')
f.close()