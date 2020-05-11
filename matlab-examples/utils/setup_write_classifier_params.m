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

function setup_write_classifier_params(fid,classifier_params,title)
%This function prints part of the setup.py file that will be read from the python
%scripting engine. In particular, it prints the parameters related to the
%classifiers.

fprintf(fid,[ title ' = dotdictify({\n']);

if(strcmp(classifier_params.classifier,'SVM_poly'))
    fprintf(fid,'\t''SVM_poly'': {\n');
    fprintf(fid,'\t\t''lib'': ''sklearn.svm.SVC'',\n');
    fprintf(fid,'\t\t''common'': {''kernel'': ''poly''},\n');
    fprintf(fid,'\t\t''grid_search'': {''param_grid'': dict(gamma=np.logspace(-6, -1, 10))},\n');
    fprintf(fid,'\t},\n');

elseif(strcmp(classifier_params.classifier,'SVM_rbf'))
     fprintf(fid,'\t''SVM_rbf'': {\n');
        fprintf(fid,'\t\t''lib'': ''sklearn.svm.SVC'',\n');
        fprintf(fid,'\t\t''common'': {''kernel'': ''rbf''},\n');
        if(classifier_params.xval==1)
            fprintf(fid,'\t\t''grid_search'': {''param_grid'': dict(C=np.logspace(-3, 2, 6), gamma=np.logspace(-3, 3, 7))},\n');
        else
            fprintf(fid,['\t\t''grid_search'': {''param_grid'': dict(C=[' num2str(classifier_params.svm_C) ', ' num2str(classifier_params.svm_C) '] , gamma=[' num2str(classifier_params.svm_gamma) '])},\n']);
        end
     fprintf(fid,'\t},\n');
                     
elseif(strcmp(classifier_params.classifier,'SVM_lin'))        
    fprintf(fid,'\t''SVM_lin'': {\n');
        fprintf(fid,'\t\t''lib'': ''sklearn.svm.SVC'',\n');
        fprintf(fid,'\t\t''common'': {''kernel'': ''linear''},\n');
    if(classifier_params.xval==1)
        fprintf(fid,'\t\t''grid_search'': {''param_grid'': dict(C=np.logspace(-3, 2, 6))},\n');
    else
        fprintf(fid,['\t\t''grid_search'': {''param_grid'': dict(C=[' num2str(classifier_params.svm_C) ', ' num2str(classifier_params.svm_C) '])},\n']);
    end
    fprintf(fid,'\t},\n');

elseif(strcmp(classifier_params.classifier,'MLP'))   
    fprintf(fid,'\t''MLP'': {\n');
        fprintf(fid,'\t\t''lib'': ''prlib.classifier.MLP'',\n');
        fprintf(fid,'\t\t''common'': {''activation_function_output'': libfann.SIGMOID_SYMMETRIC_STEPWISE,\n');
            fprintf(fid,['\t\t\t''activation_steepness_hidden'': ' num2str(classifier_params.mlp_steepness) ',\n']);
            fprintf(fid,['\t\t\t''activation_steepness_output'': ' num2str(classifier_params.mlp_steepness) ',\n']);
            fprintf(fid,'\t\t\t''iterations_between_reports'': 2500,\n');
            fprintf(fid,'\t\t\t''max_iterations'': 500},\n');
        fprintf(fid,['\t\t''grid_search'': {''param_grid'': dict(num_neurons_hidden=[' num2str(classifier_params.neurons) ', ' num2str(classifier_params.neurons) '])},\n']);
    fprintf(fid,'\t},\n');
end

fprintf(fid,'})\n\n');