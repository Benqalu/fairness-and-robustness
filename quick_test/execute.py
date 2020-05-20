import os
from time import sleep

datas=['compas','german','adult']
attrs={'adult':['race','sex'],'compas':['race','sex'],'german':['sex','age']}

for t in range(0,23):
	for data in datas:
		for attr in attrs[data]:
			cmd='python run.py %s %s'%(data,attr)
			os.system(cmd)
			sleep(3)