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

function [false_negatives,false_positives, false_negatives_copy, attack_scores, attack_scores_copy] = generate_false_negatives(classifier_params,experiment_name,exp_params,split_num,exp_folder,perfect,samples_used,num_rep)

%Generate false negatives, false positives and scores for one iterations
%(one split or repetition).
if(nargin < 6) %Used for perfect knowledge
    perfect = 1;
    samples_used = nan;
    num_rep = nan;
    false_negatives_copy = nan;
end
scores = load([exp_folder '/results/' experiment_name '/tests/base/' classifier_params.classifier '.' exp_params.dataset_name '.' num2str(split_num) '.ts.txt']);
y = scores(:,2); %Load true labels from score file
neg_scores = scores(y==-1);
pos_scores = scores(y==1);

%Calculate threshold for false positives
desired_fp=0.005;
[fp_roc tp_roc th_roc] = ROC(neg_scores,pos_scores);

[~, th] = FNatFP(fp_roc,tp_roc,th_roc,desired_fp);


if(perfect)
    attack_scores = load_attack_patterns([exp_folder '/results/' experiment_name '/tests/attack/gradient_descent/targeted_' classifier_params.classifier '.' exp_params.dataset_name '.' num2str(split_num) '.ts.txt']);
else
    attack_scores = load_attack_patterns([exp_folder '/results/' experiment_name '/tests/attack/gradient_descent/targeted_' classifier_params.classifier '-' num2str(samples_used) '-' num2str(num_rep) '@' classifier_params.classifier '.' exp_params.dataset_name '.' num2str(split_num) '.ts.txt']);
end

%For each iteration, calculate false positives and negatives
false_positives = sum(neg_scores >= th)/numel(neg_scores);
false_negatives = sum(attack_scores < th)/size(attack_scores,1);


if(~perfect) %Calculate scores and false negatives of the copy classifier
    attack_scores_copy = load_attack_patterns([exp_folder '/results/' experiment_name '/tests/attack/gradient_descent/surrogate_' classifier_params.classifier '-' num2str(samples_used) '-' num2str(num_rep) '@' classifier_params.classifier '.' exp_params.dataset_name '.' num2str(split_num) '.ts.txt']);
    false_negatives_copy = sum(attack_scores_copy < th)/size(attack_scores_copy,1);
end



end

