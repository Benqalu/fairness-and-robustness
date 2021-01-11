import sympy as sp
from math import exp

e=sp.Symbol('e')

def sigmoid(x):
	return 1.0/(1.0+sp.exp(-x))

n=2
x=[sp.Symbol(f'x_{i}') for i in range(0,3)]
s=[sp.Symbol(f's_{i}') for i in range(0,3)]
w=sp.Symbol('w')

u1=sum([sigmoid(w*x[i])*(1-s[i]) for i in range(0,n)])
v1=sum([(1-s[i]) for i in range(0,n)])

u2=sum([sigmoid(w*x[i])*s[i] for i in range(0,n)])
v2=sum([s[i] for i in range(0,n)])

P=(u1/v1+u2/v2)**2

dw=sp.diff(P,w)

print(dw)

sp.init_printing()

# print(sp.simplify(dw))