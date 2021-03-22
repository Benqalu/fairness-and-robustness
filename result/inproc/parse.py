import gzip, json
import numpy as np

def calc_angle(v1, v2, rad=False):
	u1 = np.array(flatten(v1))
	u2 = np.array(flatten(v2))
	u1 = u1 / np.linalg.norm(u1)
	u2 = u2 / np.linalg.norm(u2)
	dot = np.dot(u1, u2)
	angle = np.arccos(dot)
	if rad:
		return angle
	else:
		degree = (angle / np.pi) * 180
		return degree

def flatten(a):
	ret = []
	for item in a:
		if type(item) is list:
			ret.extend(flatten(item))
		else:
			ret.append(item)
	return ret

f=gzip.open('RnF.txt.gz','rt')
lines=f.readlines()
f.close()
print(len(lines))

data = json.loads(lines[-1])

print(type(data['wF']))
exit()

n=len(data['epoch'])
for i in range(0,n):
	# gf=data['grad_f'][i]
	# gr=data['grad_r'][i]
	# print(calc_angle(gf,gr))
	print(data['test_metric'][i])

