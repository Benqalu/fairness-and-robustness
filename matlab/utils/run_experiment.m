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

function run_experiment(set_params,setup_folder,exp_folder_name)


%Generate setup parameters
eval(fileread(set_params))

%Preparing folders for the experiments

%Save parameters file
savefile = [exp_name, '.mat'];
save(savefile);

%If main experiment folder exists, remove it
if (exist([setup_folder '/' exp_folder_name '/results/' exp_name],'dir') == 7)    
    cmd = ['rm' ' -rf ' setup_folder '/' exp_folder_name '/results/' exp_name];
    system(cmd);
end

%If figures are already saved, remove them
if (exist([setup_folder '/' exp_folder_name '/' exp_params.fig_folder],'dir') == 7)    
    cmd = ['rm' ' -rf ' setup_folder '/' exp_folder_name '/' exp_params.fig_folder];
    system(cmd);
end

%Create experiment folder
if (exist([setup_folder '/' exp_folder_name '/results/'],'dir') ~= 7)
cmd = ['mkdir ' setup_folder '/' exp_folder_name '/results/'];
system(cmd);
end

cmd = ['mkdir ' setup_folder '/' exp_folder_name '/results/' exp_name '/'];
system(cmd);

%Move parameters files to the experiment folder
cmd = ['mv setup.py ' setup_folder '/' exp_folder_name '/results/' exp_name '/'];
system(cmd);
cmd = ['mv ' exp_name '.mat ' setup_folder '/' exp_folder_name '/results/' exp_name '/'];
system(cmd);
cmd = ['mkdir ' setup_folder '/' exp_folder_name '/' exp_params.fig_folder '/'];
system(cmd);
%%%%%%%%%%%%%%%%%%%%%

%RUN EXPERIMENT

%%%%%%%%%%%%%%%%%%%%%
cmd = ['python ' exp_params.code_folder '/runexp.py ' setup_folder '/' exp_folder_name '/results/' exp_name];
system(cmd);

