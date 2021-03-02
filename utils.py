import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calc_angle(v1,v2,rad=False):
	u1=v1/np.linalg.norm(v1)
	u2=v2/np.linalg.norm(v2)
	dot=np.dot(u1,u2)
	angle=np.arccos(dot)
	if rad:
		return angle
	else:
		degree=(angle/np.pi)*180
		return degree

def get_data(data,attr,binary=False):
	if binary:
		df=pd.read_csv(f'./data/{data}_binary.csv')
	else:
		df=pd.read_csv(f'./data/{data}.csv')
	metadata=df.to_numpy()
	metaattr=list(df.columns)

	metadata=np.array(metadata,dtype=float)

	if not binary:
		scaler=MinMaxScaler()
		metadata=scaler.fit_transform(metadata)

	s=metadata[:,metaattr.index(attr)].reshape(-1)
	y=metadata[:,-1].reshape(-1)
	X=metadata[:,:-1]
	X=np.delete(X,metaattr.index(attr),axis=1)
	X=np.hstack([s.reshape(-1,1),X])
	return X,y.reshape(-1,1)

if __name__=='__main__':
	print(calc_angle([-1,1],[0,10]))

