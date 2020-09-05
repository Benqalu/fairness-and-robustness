import os,sys,argparse,subprocess
from time import sleep
from multiprocessing import Pool
from parallel import Parallel

datas=['compas','german','adult']
attrs={'adult':['race','sex'],'compas':['race','sex'],'german':['sex','age']}

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, default=-1)
args = parser.parse_args()

def exec(cmd):
	print("Running:", cmd)
	subprocess.run(cmd.strip().split(' '), capture_output=False)

if args.p!=-1:
	print(f'Parallel computing with {args.p} threads...')

	if 1==1:
		combs=[]
		for t in range(0,23):
			for sens in ['','-s']:
				for data in datas:
					for attr in attrs[data]:
							for tran in ['RW','OP']:
								combs.append('python run.py -d %s -a %s -f %s %s'%(data,attr,tran,sens))
		pool=Pool(args.p)
		pool.map(exec,combs)
		pool.join()
	else:
		pool=Parallel(p=args.p)
		for t in range(0,23):
			combs=[]
			for sens in ['','-s']:
				for data in datas:
					for attr in attrs[data]:
							for tran in ['RW','OP']:
								combs.append('python run.py -d %s -a %s -f %s %s'%(data,attr,tran,sens))
			pool.add_cmd(combs)
			pool.run(info=True, shell=True)
else:
	for item in combs:
		subprocess.call(item)
		sleep(3)
