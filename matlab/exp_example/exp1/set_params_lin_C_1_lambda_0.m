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

%This are the parameters generated for the experiment. They will be saved
%in a file .mat with the same name of the exp_name variable. Please note
%that this file is generated to ensure that the correct parameters will be
%loaded when the plot is generated
%NOTE: TO REPLICATE THE EXPERIMENTS OF THE PAPER, THE VALUE OF
%SVM_C,SVM_GAMMA AND NEURONS MUST BE THE SAME BETWEEN EVIL CLASSIFIER AND
%NORMAL CLASSIFIER



exp_params.dataset_name = 'norm_pdf_med';
exp_params.test_fraction = 0.1;
exp_params.nfolds = 3;
exp_params.nsplits = 1;
exp_params.threads = 1;
exp_params.norm_weights = 'norm_weights.txt';
exp_params.main_folder = './data_exp'; %In this folder the experimental data will be memorized
exp_params.fig_folder = 'fig'; % In this folder (inside the experiment folder) the pictures will be saved
exp_params.code_folder = '../adversariaLib'; % This is the relative path to the adversarialib code folder
exp_params.dataset_folder = '../../../../../dataset'; %This is the folder in which the dataset is stored


classifier_params.classifier = 'SVM_lin'; % 'SVM_rbf' or 'SVM_lin'
classifier_params.xval=0;
classifier_params.mlp_steepness = 0.0005;
classifier_params.neurons = 5;
classifier_params.svm_C = 1;
classifier_params.svm_gamma = 1;

evil_classifier_params.training_size = 100; 
evil_classifier_params.num_rep = 1; %Number of classifier's copies to learn by randomly sampling a training set
evil_classifier_params.classifier = classifier_params.classifier;
evil_classifier_params.xval= classifier_params.xval;
evil_classifier_params.mlp_steepness = classifier_params.mlp_steepness;
evil_classifier_params.neurons = classifier_params.neurons;
evil_classifier_params.svm_C = classifier_params.svm_C;
evil_classifier_params.svm_gamma = classifier_params.svm_gamma;

mimicry_params.mimicry_distance = 'kde_hamming';
mimicry_params.lambda = 0;
mimicry_params.kde_gamma = 0.1;
mimicry_params.max_leg_patterns = 100;

constraint_params.constraint = 'only_increment'; %Choose between: box, hamming, only_increment
constraint_params.constraint_step = 5;
constraint_params.max_boundaries = 10;

plot_params.title=['Example' ...
        ', \lambda=' num2str(mimicry_params.lambda)]; %Set up plot title
plot_params.legend_lk = ['SVM LIN' ' - LK' ' (C=' num2str(classifier_params.svm_C) ')']; %Set up legend for lk
plot_params.legend_pk = ['SVM LIN' ' - PK' ' (C=' num2str(classifier_params.svm_C) ')']; %Set up legend for pk
plot_params.color = 'r';

gradient_params.grad_step = 0.01;
gradient_params.maxiter = 1000;

setup(exp_params,classifier_params,evil_classifier_params,mimicry_params,gradient_params,constraint_params);

dMAX = 0:constraint_params.constraint_step:constraint_params.constraint_step*constraint_params.max_boundaries;%Max number of feature changes

if (strcmp(classifier_params.classifier,'SVM_rbf'))
   C_gamma_neurons = classifier_params.svm_gamma;
end
        
if (strcmp(classifier_params.classifier,'SVM_lin'))
   C_gamma_neurons = classifier_params.svm_C;
end
        
if (strcmp(classifier_params.classifier,'MLP'))
   C_gamma_neurons = classifier_params.neurons;
end

% Define the experiment name. The name is composed by:
% i) classifier name ii) constraint step iii) max boundaries iv) training
% size v) lambda value (no mimicry -> lambda=0) vi) number of splits vii)
% number of repetitions viii) value of gamma, C or neurons dependent on the
% type of classifier.
exp_name = ['exp_' classifier_params.classifier '_' ...
    num2str(constraint_params.constraint_step) '_' ...
    num2str(constraint_params.max_boundaries) '_' ...
    num2str(evil_classifier_params.training_size) '_' ...
    num2str(mimicry_params.lambda) '_' ...
    num2str(exp_params.nsplits) '_' ...
    num2str(evil_classifier_params.num_rep) '_' ...
    num2str(C_gamma_neurons)];



