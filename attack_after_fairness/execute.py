import os,sys,argparse,subprocess
from time import sleep
from multiprocessing import Pool
from parallel import Parallel

datas=['compas','german','adult']
attrs={'adult':['race','sex'],'compas':['race','sex'],'german':['sex','age']}

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, default=-1)
args = parser.parse_args()

combs=[]
for t in range(0,23):
	for sens in ['','-s']:
		for data in datas:
			for attr in attrs[data]:
					for tran in ['RW','OP']:
						combs.append('python run.py -d %s -a %s -f %s %s'%(data,attr,tran,sens))

def exec(args):
	cmd='python run.py -d %s -a %s %s -f %s'%item
	subprocess.call(cmd)
	sleep(3)

if args.p!=-1:
	print(f'Parallel computing with {args.p} threads...')
	pool=Parallel(p=args.p)
	pool.add_cmd(combs)
	pool.run(info=True, shell=True)
else:
	for item in combs:
		subprocess.call(item)
		sleep(3)
