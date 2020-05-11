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

function setup(exp_params,classifier_params,evil_classifier_params,mimicry_params,gradient_params,constraint_params)
%This function prints the whole setup.py file (for the classifier
%parameters, it calls setup_write_classifier.m)

fid = fopen('setup.py','w');

fprintf(fid,'import numpy as np\n');
fprintf(fid,'from os.path import realpath, dirname, join, abspath\n');
fprintf(fid,'from pyfann import libfann\n');
fprintf(fid,'from util import dotdictify, BASE\n\n');

%===============================================================================
% Dataset and Experiment name
%===============================================================================
fprintf(fid,'EXP_PATH = dirname(realpath(__file__))\n');
fprintf(fid,['DSET_FOLDER = abspath(join(EXP_PATH, ''' exp_params.dataset_folder '''))\n']);
fprintf(fid,['DSET_NAME = ''' exp_params.dataset_name '''\n\n' ]);
fprintf(fid,['TEST_FRACTION = ' num2str(exp_params.test_fraction) '\n']);
fprintf(fid,['NFOLDS = ' num2str(exp_params.nfolds) '\n']);
fprintf(fid,['NSPLITS = ' num2str(exp_params.nsplits) '\n']);



setup_write_classifier_params(fid,classifier_params,'CLASSIFIER_PARAMS');


fprintf(fid,'GRID_PARAMS = dotdictify({''iid'': True, ''n_jobs'': 1})\n\n');

if(~isempty(exp_params.norm_weights))
    fprintf(fid,['NORM_WEIGHTS_FILEPATH = join(DSET_FOLDER, ''' exp_params.norm_weights ''')\n']);
    fprintf(fid,'NORM_WEIGHTS = np.array([float(item) for item in open(NORM_WEIGHTS_FILEPATH).read().split()])\n\n');
end


%===============================================================================
% Attack Parameters
%===============================================================================

setup_write_classifier_params(fid,evil_classifier_params,'SURROGATE_CLASSIFIER_PARAMS');

fprintf(fid,'ATTACK_PARAMS = dotdictify({\n');
    fprintf(fid,'\t''gradient_descent'': {\n');
        fprintf(fid,['\t\t''attack_class'': 1, ''maxiter'': ' num2str(gradient_params.maxiter) ' ,\n']); 
        fprintf(fid,['\t\t''score_threshold'': 0, ''step'':' num2str(gradient_params.grad_step) ',\n']);
        fprintf(fid,'\t\t''fname_metachar_attack_vs_target'': ''@'',\n');
        fprintf(fid,'\t\t''fname_metachar_samples_repetitions'': ''-'',\n');
       
        
        if(strcmp(constraint_params.constraint,'box'))
            fprintf(fid,['\t\t''constraint_function'': ''box'', ''constraint_params'': {''box_step'': ' num2str(constraint_params.constraint_step) ' },\n']);
        elseif(strcmp(constraint_params.constraint,'box_fixed'))
            fprintf(fid,['\t\t''constraint_function'': ''box_fixed'', ''constraint_params'': {''lb'': ' num2str(constraint_params.lb) ', ''ub'': ' num2str(constraint_params.ub) ' },\n']);
        elseif(strcmp(constraint_params.constraint,'only_increment'))
            fprintf(fid,['\t\t''constraint_function'': ''only_increment'', ''constraint_params'': {''only_increment_step'': ' num2str(constraint_params.constraint_step) ', ''feature_upper_bound'': 1},\n']); 
        elseif(strcmp(constraint_params.constraint,'hamming'))
            fprintf(fid,'\t\t''constraint_function'': ''hamming'', ''constraint_params'': {},\n'); 
        else
            fprintf(fid,'\t\t''constraint_function'': None, ''constraint_params'': {},\n'); 
        end
        
        fprintf(fid,['\t\t''stop_criteria_window'': 20, ''stop_criteria_epsilon'': 10**(-9), ''max_boundaries'': ' num2str(constraint_params.max_boundaries) ' ,\n']);
        fprintf(fid,['\t\t''lambda_value'':' num2str(mimicry_params.lambda) ', ''mimicry_distance'': ''' mimicry_params.mimicry_distance ''', ''relabeling'': True,\n']);
        fprintf(fid,['\t\t''mimicry_params'' : { ''max_leg_patterns'': ' num2str(mimicry_params.max_leg_patterns) ' , ''gamma'': ' num2str(mimicry_params.kde_gamma) '},\n']); 
        fprintf(fid,['\t\t''save_attack_patterns'': True, ''threads'': ' num2str(exp_params.threads) ',\n']);
        fprintf(fid,'\t\t''training'': {\n');
            fprintf(fid,['\t\t\t''dataset_knowledge'': {''samples_range'': range(' num2str(evil_classifier_params.training_size) ' ,' num2str(2*evil_classifier_params.training_size)  ',' num2str(evil_classifier_params.training_size) '), ''repetitions'': ' num2str(evil_classifier_params.num_rep) '},\n']);
        fprintf(fid,'\t\t},\n'); 
    fprintf(fid,'\t},\n');
fprintf(fid,'})\n\n');

fclose(fid);
return;
             
             
             