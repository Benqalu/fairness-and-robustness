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

function experiment_name = run_plot(fig,exp_folder,classifier_params,constraint_params,evil_classifier_params,exp_params,mimicry_params,lambda,C_gamma_neurons,plot_params,plot_number)

%THIS IS THE FUNCTION FOR PLOTTING EXPERIMENTS
%IT PLOTS BOTH PERFECT AND IMPERFECT KNOWLEDGE.


% Define the experiment name. The name is composed by:
% i) classifier name ii) constraint step iii) max boundaries iv) training
% size v) lambda value (no mimicry -> lambda=0) vi) number of splits vii)
% number of repetitions viii) value of gamma, C or neurons dependent on the
% type of classifier.

    experiment_name = ['exp_' classifier_params.classifier '_' ...
        num2str(constraint_params.constraint_step) '_' ...
        num2str(constraint_params.max_boundaries) '_' ...
        num2str(evil_classifier_params.training_size) '_' ...
        num2str(lambda) '_' ...
        num2str(exp_params.nsplits) '_' ...
        num2str(evil_classifier_params.num_rep) '_' ...
        num2str(C_gamma_neurons)];

%Check the integrity of the experiment
integrity = [exp_folder '/results/'  experiment_name '/' experiment_name '.mat'];
load(integrity);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Obtain scores - Prepare for plots
%falneg_perfect = false negatives for perfect knowledge
%falneg_lim = false negatives for limited knowledge
%falpos_perfect = false positives for perfect knowledge
%falpos_lim = false positives for limited knowledge

[falneg_perfect,falpos_perfect,falneg_lim,falneg_lim_copy,attack_scores] = merge_false_negatives(classifier_params,dMAX,evil_classifier_params,exp_params,experiment_name,exp_folder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Plot false negatives 

fig = plot_false_negatives(fig,classifier_params,constraint_params, mimicry_params,dMAX,falneg_perfect,falneg_lim,plot_params);



end
