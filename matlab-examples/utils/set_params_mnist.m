%            _                               _       _     _ _     
%   __ _  __| |_   _____ _ __ ___  __ _ _ __(_) __ _| |   (_) |__  
%  / _` |/ _` \ \ / / _ \ '__/ __|/ _` | '__| |/ _` | |   | | '_ \ 
% | (_| | (_| |\ V /  __/ |  \__ \ (_| | |  | | (_| | |___| | |_) | 
%  \__,_|\__,_| \_/ \___|_|  |___/\__,_|_|  |_|\__,_|_____|_|_.__/  
% 
% adversariaLib - Advanced library for the evaluation of machine 
% learning algorithms and classifiers against adversarial attacks.
% 
% Copyright (C) 2013, Igino Corona, Battista Biggio, Davide Maiorca, 
% Dept. of Electrical and Electronic Engineering, University of Cagliari, Italy.
% 
% adversariaLib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% adversariaLib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%    MATLAB SCRIPT for generating setup.py        %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   This are the parameters generated for the experiment                  %
%   from which setup.py will be automatically saved.                      %
%   They will be saved in a MAT file with the same name as 'exp_name'.    %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



exp_params.dataset_name = 'data';
exp_params.test_fraction = 0.1; %fraction of randomly chosen test data (the rest will be used for training)
exp_params.nfolds = 3; %number of folds for cross validation
exp_params.nsplits = 1; %number of splits
exp_params.threads = 1; %number of threads (note: if you set -1 it will use all the processors)
exp_params.norm_weights = ''; %normalization parameters for the PDF data.
exp_params.fig_folder = 'fig'; %in this folder (inside each experiment folder) the plot will be saved
exp_params.code_folder = '../adversariaLib'; % This is the relative path to the adversarialib code folder
exp_params.dataset_folder = 'expMNIST/data'; %This is the folder in which the dataset is stored

classifier_params.classifier = classifier; % 'SVM_rbf' or 'MLP'
classifier_params.xval=0; 
classifier_params.mlp_steepness = 0.0005;
classifier_params.neurons = 3;
classifier_params.svm_C = C;
classifier_params.svm_gamma = Gamma;

evil_classifier_params.training_size = 10; 
evil_classifier_params.num_rep = 1; %Number of classifier's copies to learn by randomly sampling a training set
evil_classifier_params.classifier = classifier_params.classifier;
evil_classifier_params.xval= classifier_params.xval;
evil_classifier_params.mlp_steepness = classifier_params.mlp_steepness;
evil_classifier_params.neurons = classifier_params.neurons;
evil_classifier_params.svm_C = classifier_params.svm_C;
evil_classifier_params.svm_gamma = classifier_params.svm_gamma;

mimicry_params.mimicry_distance = 'kde_euclidean';
mimicry_params.lambda = lambda; %Modify lambda > 0 to enforce mimicry in our gradient-based evasion attack (see ECML paper)
mimicry_params.kde_gamma = 0.2;
mimicry_params.max_leg_patterns = 10;

constraint_params.constraint = 'box_fixed';   % 'box', 'only_increment', 'hamming'
constraint_params.lb=0;
constraint_params.ub=1;
constraint_params.max_boundaries = 1;
constraint_params.constraint_step = 1; 


gradient_params.grad_step = grad_step;
gradient_params.maxiter = maxiter;

setup(exp_params,classifier_params,evil_classifier_params,mimicry_params,gradient_params,constraint_params);

dMAX = 0:constraint_params.constraint_step:constraint_params.constraint_step*constraint_params.max_boundaries; %max distance in feature space



