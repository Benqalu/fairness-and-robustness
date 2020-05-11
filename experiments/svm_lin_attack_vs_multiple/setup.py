import numpy as np
from os.path import realpath, dirname, join, abspath
from pyfann import libfann
from util.vars import dotdictify, BASE

EXP_PATH = dirname(realpath(__file__))
DSET_FOLDER = abspath(join(EXP_PATH, '../../dataset'))
DSET_NAME = 'norm_pdf_med'

TEST_FRACTION = 0.15
NFOLDS = 3
NSPLITS = 3
CLASSIFIER_PARAMS = dotdictify({
	'SVM_lin': {
		'lib': 'sklearn.svm.SVC',
		'common': {'kernel': 'linear'},
		'grid_search': {'param_grid': dict(C=np.logspace(-3, 2, 6))},
	},
	'SVM_rbf': {
		'lib': 'sklearn.svm.SVC',
		'common': {'kernel': 'rbf'},
		'grid_search': {'param_grid': dict(C=np.logspace(-3, 2, 6), gamma=np.logspace(-3, 3, 7))},
	},
	'SVM_poly': {
		'lib': 'sklearn.svm.SVC',
		'common': {'kernel': 'poly'},
		'grid_search': {'param_grid': dict(C=np.logspace(-3, 2, 6), gamma=np.logspace(-3, 3, 7))},
	},
	'MLP': {
		'lib': 'prlib.classifier.MLP',
		'common': {'activation_function_output': libfann.SIGMOID_SYMMETRIC_STEPWISE,
			'activation_steepness_hidden': 0.001,
			'activation_steepness_output': 0.001,
			'iterations_between_reports': 500,
			'max_iterations': 5000},
		'grid_search': {'param_grid': dict(num_neurons_hidden=[3,5,7])},
	},
})

GRID_PARAMS = dotdictify({'iid': True, 'n_jobs': 1})

NORM_WEIGHTS_FILEPATH = join(DSET_FOLDER, 'norm_weights.txt')
NORM_WEIGHTS = np.array([float(item) for item in open(NORM_WEIGHTS_FILEPATH).read().split()])

SURROGATE_CLASSIFIER_PARAMS = dotdictify({
	'SVM_lin': {
		'lib': 'sklearn.svm.SVC',
		'common': {'kernel': 'linear'},
		'grid_search': {'param_grid': dict(C=np.logspace(-3, 2, 6))},
	},
})

ATTACK_PARAMS = dotdictify({
	'gradient_descent': {
			'attack_class': 1, 'maxiter': 500,
			'score_threshold': 0, 'step':0.01,
			'fname_metachar_attack_vs_target': '@',
			'fname_metachar_samples_repetitions': '-',
			'constraint_function': 'only_increment', 'constraint_params': {'only_increment_step': 1},
			'stop_criteria_window': 20, 'stop_criteria_epsilon': 10**(-9), 'max_boundaries': 10,
			'lambda_value':10, 'mimicry_distance': 'kde_hamming', 'relabeling': True,
			'mimicry_params' : {'max_leg_patterns': 100 , 'gamma': 0.001},
			'save_attack_patterns': True, 'threads': 1,
			'training': {
				'dataset_knowledge': {'samples_range': range(50,100,50), 'repetitions': 1},
			},
	},
})