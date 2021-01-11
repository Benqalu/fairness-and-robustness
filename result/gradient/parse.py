from matplotlib import pyplot as plt

def hist(ax, data='adult',attr='sex'):
	angles_offset=[]
	angles_switch=[]
	f=open('angles_%s_%s.txt'%(data,attr))
	for line in f:
		line=line.replace('nan','-1.0')
		line=eval(line)
		angles_offset.append(line[0])
		angles_switch.append(line[2])
	f.close()

	bins=[i*5 for i in range(6,31)]
	ax.hist(angles_offset,bins=bins,alpha=0.4,label='offset')
	ax.hist(angles_switch,bins=bins,alpha=0.4,label='flip')

	ax.legend()

	ax.set_xticks([i*10 for i in range(3,16)])
	ax.set_title('angles_%s_%s'%(data,attr))
	ax.set_xlabel('Angle of gradients')

fig,axs=plt.subplots(3,2)

datas=['adult','compas','hospital']
attrs=['race','sex']

for i in range(0,3):
	for j in range(0,2):
		hist(axs[i][j],data=datas[i],attr=attrs[j])

plt.tight_layout()
plt.show()