import torch
import numpy as np
from utils import load_split
from TorchAdversarial import TorchAdversarial


def test(data,attr):
	train, test = load_split(data, attr)
	model = TorchAdversarial(hiddens=[128], hidden_activation=torch.nn.Tanh)
	model.fit(train['X'], train['y'])
	print(model.metrics(test['X'], test['y'], s=test['s']))

if __name__=='__main__':
	test('adult','race')