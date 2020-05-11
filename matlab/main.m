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

%This is the Matlab interface and plot-generation code for AdversariaLib.

clc
clear all
close all

addpath utils
addpath exportfig
addpath roc

tic 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%                     PARAMETERS                  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set path to 'python', 'pyfann', and dynamic libraries
%Mac OS X: they should be respectively located
%          in /usr/opt/bin (macports) and /usr/local/bin
%LINUX: you can typically find them in /usr/bin and /usr/local/bin
setenv(      'PATH',       '/opt/local/bin:/usr/bin:/bin:/usr/local/bin');
setenv('DYLD_LIBRARY_PATH','/opt/local/bin:/usr/bin:/bin:/usr/local/bin');

%folder which contains the exp* folders for each experiment
setup_folder = 'exp_example'; %a simple example (quick experiment - it will take just some minutes)

%uncomment the line below to run the complete set of exp. of our paper.
%setup_folder = 'exp_paper_ecml'; 


make_exp = 1;   %run experiments from scratch (execute python scripts)
make_plots = 1; %create plots based on stored results (needs graphical interface open -- no matlab from command line)



%figure properties
fontsize = 14;
width = 15;
height = 8;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




exp_folders = dir([setup_folder '/exp*']); %retrieve experiment folders

%run experiments for each experiment folder (execute python script).
%Results will be saved in the corresponding exp* folder (by default, inside 'results/')
if make_exp
    for i=1:numel(exp_folders)

       setup_files = dir([setup_folder '/' exp_folders(i).name '/set*']);
       for j=1:numel(setup_files)
            run_experiment([setup_folder '/' exp_folders(i).name '/' setup_files(j).name],setup_folder, exp_folders(i).name);
       end

    end
end

%make plots for each experiment folder. The plot will be saved in the
%corresponding exp* folder (by default, inside 'fig/')
if make_plots
    for i=1:numel(exp_folders)
        [fig opts]=createfig(fontsize,width,height);
        setup_files = dir([setup_folder '/' exp_folders(i).name '/set*']);
        C_or_gamma_or_neurons = [];
           for j=1:numel(setup_files)
               exp_folder = [setup_folder '/' exp_folders(i).name];
               eval(fileread([setup_folder '/' exp_folders(i).name '/' setup_files(j).name]));
               if strcmp(classifier_params.classifier, 'MLP')
                   C_or_gamma_or_neurons = classifier_params.neurons;
               end

               if strcmp(classifier_params.classifier, 'SVM_rbf')
                   C_or_gamma_or_neurons = classifier_params.svm_gamma;
               end

               if strcmp(classifier_params.classifier, 'SVM_lin')
                   C_or_gamma_or_neurons = classifier_params.svm_C;
               end

               experiment_name =  run_plot(fig,exp_folder,classifier_params,constraint_params,evil_classifier_params,exp_params,mimicry_params,mimicry_params.lambda,C_or_gamma_or_neurons,plot_params);
           end
       exportfig(fig, [setup_folder '/' exp_folders(i).name '/' exp_params.fig_folder '/'  'plot.eps'],opts);
       eps2pdf([setup_folder '/' exp_folders(i).name '/' exp_params.fig_folder])  
    end
end

toc
disp('Program terminated.');
