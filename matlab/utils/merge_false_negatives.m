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

function [falneg_perfect, falpos_perfect, falneg_lim, falneg_lim_copy, attack_scores, attack_scores_cp_real, attack_scores_cp] = merge_false_negatives(classifier_params,dMAX,evil_classifier_params,exp_params,experiment_name,exp_folder)

%This function returns the false negatives and positives for perfect and
%limited knowledge for ALL runs

for s=0:exp_params.nsplits-1 %Perfect Knowledge case
    [falneg_perfect(s+1,:),falpos_perfect(s+1),~,attack_scores{s+1}] = generate_false_negatives(classifier_params,experiment_name,exp_params,s,exp_folder);
end

i = 0;
for r=0:evil_classifier_params.num_rep-1 %Limited Knowledge case
    for s=0:exp_params.nsplits-1
        i = i+1;
        [falneg_lim(i,:),falpos_lim(i), falneg_lim_copy(i,:) attack_scores_cp_real{i} attack_scores_cp{i}] = generate_false_negatives(classifier_params,experiment_name,exp_params,s,exp_folder,0,evil_classifier_params.training_size,r);

    end
end

