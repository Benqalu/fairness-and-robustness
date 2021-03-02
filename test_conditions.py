
def conf(y_hat,y):
	ret=''
	if y_hat==y:
		ret+='T'
	else:
		ret+='F'
	if y_hat<0:
		ret+='N'
	else:
		ret+='P'
	return ret

def case1():
	ret=set([])
	for d in [-1,1]:
		for s in [-1,1]:
			if (d>0 and s<0) or (d<0 and s>0):
				promoted=True
			else:
				promoted=False
			for y_hat in [-1,1]:
				for y in [-1,1]:
					orig=conf(y_hat,y)
					dest=conf(y_hat,-y)
					flip=False
					if promoted and (orig=='TP' or dest=='FN'):
						flip=True
					if not promoted and (dest=='TP' or orig=='FN'):
						flip=True
					condi='d=%+d, s=%+d, p=%s, y_hat=%+d, y=%+d, %s-->%s'%(d,s,str(promoted)[0],y_hat,y,orig,dest)
					if flip:
						ret.add(condi)
	return ret


def case2():
	ret=set([])
	for d in [-1,1]:
		for s in [-1,1]:
			if (d>0 and s<0) or (d<0 and s>0):
				promoted=True
			else:
				promoted=False
			for y_hat in [-1,1]:
				for y in [-1,1]:
					orig=conf(y_hat,y)
					if y_hat!=y:
						dest=conf(y_hat,-y)
					else:
						dest=conf(-y_hat,-y)
					flip=False
					if promoted and (orig=='TP' or dest=='FN'):
						flip=True
					if not promoted and (dest=='TP' or orig=='FN'):
						flip=True
					condi='d=%+d, s=%+d, p=%s, y_hat=%+d, y=%+d, %s-->%s'%(d,s,str(promoted)[0],y_hat,y,orig,dest)
					if flip:
						ret.add(condi)
	return ret

flips1=case1()
flips2=case2()
for item in flips1&flips2:
	print(item)