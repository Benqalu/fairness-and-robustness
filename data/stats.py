import pandas as pd
import numpy as np

datas = ['adult', 'compas', 'hospital']

for data in datas:
	df_train = pd.read_csv(f'{data}_train.csv')
	df_test = pd.read_csv(f'{data}_test.csv')

	df = df_train.append(df_test, ignore_index=True)

	print(df.shape)


