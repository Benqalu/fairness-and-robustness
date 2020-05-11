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

% This code produces 2D plots similar to those shown in Fig. 1 of our ECML paper

clc
clear all
close all

!cat COPYRIGHT.txt

%set path to 'python', 'pyfann', and dynamic libraries
%Mac OS X: they should be respectively located
%          in /usr/opt/bin (macports) and /usr/local/bin
%LINUX: you can typically find them in /usr/bin and /usr/local/bin
setenv(      'PATH',       '/opt/local/bin:/usr/bin:/bin:/usr/local/bin');
setenv('DYLD_LIBRARY_PATH','/opt/local/bin:/usr/bin:/bin:/usr/local/bin');

addpath utils
addpath exportfig

grid_size=50;
grid_coords = [-6 6 -6 6];

folder = 'exp-2d'; %where to store results

system(['rm -rf ' folder]); %clear previous run

mkdir(folder)
mkdir([folder  '/data']);
mkdir([folder  '/exp']);
mkdir([folder  '/exp/base']);
mkdir([folder  '/exp/attack']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SETUP.PY
%

%Modify lambda > 0 to enforce mimicry in
%our gradient-based evasion attack (see ECML paper)
lambda = 0;  

classifier = 'SVM_rbf'; %'SVM_lin', 'SVM_rbf', or 'MLP'

C=1; %regularization parameter for SVM.
Gamma=0.5; %ignored for SVM_lin 

neurons=3; %hidden neurons for MLP (neural net)

grad_step=0.01;
maxiter=500;

set_params_2d %generates setup.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Sample Gaussian dataset
mu=1; sigma = 0.7; num_feat=2; n=10; %number of samples per class
[x y] = load_gaussian_data(mu,sigma,n,num_feat);

%Store dataset to txt file
dlmwrite([folder '/data/data.txt'], [x y], 'delimiter', ' ', 'precision', 3);

%Learn classifier on stored dataset     
system(['python ../adversariaLib/api/learn_classifiers.py ./ ' folder '/data/data.txt ' folder '/exp/base/']);


%calculate values of the discriminant function g(x) on the feature space
%and of the mimicry component (prob_KDE) to plot the obj. function as colored background
[x1 x2 x_test] = grid_discriminant_function(grid_coords,grid_size);
y_test = ones(size(x_test,1),1);     
dlmwrite([folder '/data/test_grid.txt'], [x_test y_test], 'delimiter', ' ', 'precision', 3);
system(['python ../adversariaLib/api/test_classifiers.py ./ ' folder '/data/test_grid.txt ' folder '/exp/base/ ' folder '/exp/']);
g_test=load([folder '/exp/scores.' classifier_params.classifier '.txt']);
g_test = g_test(:,1);
g_test = reshape(g_test, size(x1,1), size(x1,2));
prob_kde = pKDE(x,y,x_test,mimicry_params.mimicry_distance,mimicry_params.kde_gamma);
prob_kde = reshape(prob_kde, size(x1,1), size(x1,2));
obj = g_test - mimicry_params.lambda*prob_kde;

%run gradient-descent evasion on all samples of class=+1
system(['python ../adversariaLib/api/gradient_descent_attack.py' ' ./ ' ...
    folder '/data/data.txt ./' folder '/exp/base/' classifier_params.classifier  ...
    ' ./' folder '/exp/base/' classifier_params.classifier ' ./' folder '/exp/attack']);

%load attack samples (gradient trajectories)
disp('Loading attack patterns...');
x_attack = load_attack_patterns([folder '/exp/attack/attack_patterns.' classifier_params.classifier '.txt']);

fontsize=14;
w=20;
h=10;
[fig opts]=createfig(fontsize,w,h);
decision_plot(x_attack,x1,x2,obj,x,y,g_test,mimicry_params.lambda);
%plot box_fixed constraint, if set
if(strcmp(constraint_params.constraint,'box_fixed'))
    xx=constraint_params.lb:0.01:constraint_params.ub;
    plot(xx,constraint_params.lb*ones(1,numel(xx)),'k--'); hold on;
    plot(xx,constraint_params.ub*ones(1,numel(xx)),'k--');
    plot(constraint_params.lb*ones(1,numel(xx)),xx,'k--');
    plot(constraint_params.ub*ones(1,numel(xx)),xx,'k--');
end
applytofig(fig,opts);
exportfig(fig,['fig/2D-' classifier_params.classifier '-' num2str(mimicry_params.lambda) '.eps'],opts);
eps2pdf('fig');

!rm -rf setup.pyc
system(['mv setup.py ' folder '/']);
disp(['Experiment completed. The displayed plot can be found in fig/. Results are stored in ' folder '/.']);

