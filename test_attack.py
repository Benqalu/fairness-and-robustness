import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from sklearn.linear_model import LogisticRegression
from art.classifiers.scikitlearn import ScikitlearnLogisticRegression as ART_LR
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
	import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from art.attacks.evasion import ProjectedGradientDescent as ProjectedGradientDescentAttack
from art.attacks.evasion import FastGradientMethod as FastGradientMethodAttack

from metric import Metric

def to_categorical(a):
	return OneHotEncoder().fit_transform(np.array(a).reshape(-1,1)).toarray()

data=load_preproc_data_adult()
X=data.features
y=data.labels.reshape(-1)

train={}
test={}

train['X'], test['X'], train['y'], test['y'] = train_test_split(X, y, test_size=0.3)

model=ART_LR(LogisticRegression())
model.fit(train['X'], to_categorical(train['y']))

test['y_pred']=model.predict(test['X'])[:,1]

metric=Metric(true=test['y'], pred=test['y_pred'])
print(metric.accuracy())

attack=FastGradientMethodAttack(estimator=model, norm=np.inf, eps=0.1)
test['X_adv']=attack.generate(x=test['X'])

print(test['X_adv'])
