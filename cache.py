

def validate_fairness():

	from sklearn.model_selection import train_test_split

	seed=24

	for data in ['adult','compas','hospital']:
		for attr in ['race', 'sex']:

			print(data,attr)

			X,y=getdata(data,attr)

			X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

			model = FaroLR(lr=1E-3, n_epoch=3000, fairness=None, seed=24)
			epochdata = model.validate(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
			
			f=open('./result/regularizer/epoch_fairness_%s_%s_%d.txt'%(data, attr, seed),'w')
			f.write(str(epochdata)+'\n')
			f.close()

			model = FaroLR(lr=1E-3, n_epoch=3000, fairness=0.3, seed=24)
			epochdata = model.validate(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
			
			f=open('./result/regularizer/epoch_fairness_%s_%s_%d.txt'%(data, attr, seed),'a')
			f.write(str(epochdata)+'\n')
			f.close()

def collect(values):
	bins=np.zeros(20).astype(int)
	for item in values:
		if int(item/0.05)==20:
			bins[-1]+=1
		else:
			bins[int(item/0.05)]+=1
	return bins

def validate_robustness():
	seed=24

	for data in ['adult','compas','hospital']:
		for attr in ['race']:

			X,y=getdata(data,attr)

			taus=[]

			for tau in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.2, 0.3, 0.4, 0.5]:
				model = FaroLR(lr=1E-3, n_epoch=2000, robustness=tau, seed=24)
				model.fit(X,y)
				X_attack=model.attack(X,y,eps=0.1)
				acc_original=model.predict(X,y)
				acc_attacked=model.predict(X_attack,y)
				taus.append((tau, acc_original, acc_attacked))
				print('>>> %.2f, %.4f, %.4f <<<'%(tau, acc_original, acc_attacked))

			f=open('./result/regularizer/robustness_%s_%s_switch_%d.txt'%(data, attr, seed),'w')
			f.write(str(taus)+'\n')
			f.close()