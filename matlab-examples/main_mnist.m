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

folder = 'exp-mnist'; %where to store results

system(['rm -rf ' folder ]); %clear previous run

mkdir(folder)
mkdir([folder '/data'])
mkdir([folder '/exp'])
mkdir([folder '/exp/base']);
mkdir([folder '/exp/attack'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SETUP.PY
%

%Modify lambda > 0 to enforce mimicry in
%our gradient-based evasion attack (see ECML paper)
lambda = 0;  

classifier = 'SVM_lin'; %or 'SVM_rbf' 'MLP'

C=1; %regularization parameter for SVM.
Gamma=0.1; %ignored for SVM_lin 

grad_step=0.05;
maxiter=100;

set_params_mnist %generates setup.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%select numbers (positive/attack and negative/benign class)
pos_class = '3';
neg_class = '7';


[x y] = load_tr_ts_mnist('mnist-data',pos_class,neg_class);
x = x ./ 255; %normalize data in [0,1]

%take a subset of n_tr samples radomly from training data
n_tr = 100;
tr_idx = randsample(size(x,1),n_tr);
tr_idx = tr_idx(1:n_tr);
x = x(tr_idx,:);
y = y(tr_idx);

%Store dataset to txt file
dlmwrite([folder '/data/data.txt'], [x y], 'delimiter', ' ', 'precision', 2);
  
%Learn classifier on stored dataset     
system(['python ../adversariaLib/api/learn_classifiers.py ./ ' folder '/data/data.txt ' folder '/exp/base/']);

%Test classifier on stored dataset     
system(['python ../adversariaLib/api/test_classifiers.py ./ ' folder '/data/data.txt ' folder '/exp/base/ ' folder '/exp/']);
yclass = 2*(load([folder '/exp/scores.' classifier_params.classifier '.txt']) > 0)-1;
yclass = yclass(:,1);
disp(['Accuracy (training data): ' num2str(100*sum(yclass==y)/numel(y)) '%'])

%choose the first digit of class=+1 and run gradient-descent evasion
idx = find(y==1);
dlmwrite([folder '/data/attack_sample.txt'], [x(idx(1),:) y(idx(1))], 'delimiter', ' ', 'precision', 2);
system(['python ../adversariaLib/api/gradient_descent_attack.py' ' ./ ' ...
    folder '/data/attack_sample.txt ./' folder '/exp/base/' classifier_params.classifier  ...
    ' ./' folder '/exp/base/' classifier_params.classifier ' ./' folder '/exp/attack']);


%load attack samples (modified during the gradient descent)
disp('Loading attack patterns...');
x_attack = load_attack_patterns([folder '/exp/attack/attack_patterns.' classifier_params.classifier '.txt']);
x_attack = x_attack{1};

scores = load_attack_scores([folder '/exp/attack/surrogate_scores.' classifier_params.classifier '.txt']);

%Plot results
fontsize=14;
w=30;
h=12;
[fig opts]=createfig(fontsize,w,h);

subplot(1,4,1)
display_character(x_attack(1,:));
title(['Before attack (' pos_class ' vs ' neg_class ')'])
axis square

subplot(1,4,2)
idx = find(scores <= 0, 1 );
display_character(x_attack(min(idx,size(x_attack,1)),:));
title('After attack, g(x)=0')
axis square

subplot(1,4,3)
display_character(x_attack(end,:));
title('After attack, last iter.')
axis square


subplot(1,4,4)
plot(scores,'r')
hold on
plot(scores*0,'k');
title('g(x)')
grid on
axis square
xlabel('number of iterations')

axis([0 maxiter -2 2])
applytofig(fig,opts);
exportfig(fig,['fig/MNIST-' pos_class '_vs_' neg_class '-' classifier_params.classifier '-' num2str(mimicry_params.lambda) '.eps'],opts);
eps2pdf('fig');

!rm -rf setup.pyc
system(['mv setup.py ' folder '/']);
disp(['Experiment completed. The displayed plot can be found in fig/. Results are stored in ' folder '/.']);
