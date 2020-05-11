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


This README explains how to repeat the experiments of the paper "Evasion attacks against 
machine learning at test time".


====================================================

RUNNING A SIMPLE EXAMPLE

====================================================

Simply open main.m and run the script. This will launch the experiments inside the exp_example folder, which contains
the configuration file for the experiment (the template is set_params_example). You can find the
plot inside exp1/fig folder.

Be sure that the paths of python and pyfann are setup correctly. You can set them by operating on the setenv function inside
the main.m.

AdversariaLib supports advanced multi-threading. If you want more threads to be executed, you can change the
exp_params.threads value inside the configuration file and set it accordingly to the number of processors. 
If you use -1, every CPU will be used for the computations.

====================================================

We found a problem when using MATLAB on UBUNTU 13.04 for 64bit architectures (that may also occur for 32bit arch.).
In particular, the library libgfortran.so.3 is not properly loaded from MATLAB.

Here's the solution. First, locate libgfortran.so.3 as installed from the package libgfortran3

pralab@vortex$ dpkg-query -L libgfortran3
….
/usr/lib/x86_64-linux-gnu/libgfortran.so.3
….

Then, locate MATLAB's version of the same file

pralab@vortex$ locate libgfortran.so.3
...
/home/MATLAB/R2013a/sys/os/glnxa64/libgfortran.so.3
...

Finally, overwrite the MATLAB's version with the following command.
WARNING: REPLACE the first and the second path with the ones found in your case!

pralab@vortex$ sudo ln -sf /usr/lib/x86_64-linux-gnu/libgfortran.so.3 /home/MATLAB/R2013a/sys/os/glnxa64/libgfortran.so.3

====================================================

REPLICATING THE EXPERIMENTS OF OUR ECML PAPER

====================================================

Simply open main.m and set setup_folder = "exp_paper_ecml". Run the script.

If you set make_exp = 1, the whole experiment  will be executed and the results will be saved in the "results/" 
folder inside the "exp/" folders (contained in the folder "exp_ecml_paper"). 
If you have already carried out the experiments, you can set make_plots = 1 to show the plots 
(will be saved inside the "fig/" folder contained in the "exp/" folders).

Experiments can be run in background with make_exp = 1 and make_plots = 0. with the command
nohup matlab < main.m > output.txt &

The set_params is a MATLAB script that automatically generates the setup.py file run by AdversariaLib.

We already provide our computed results as a compressed archive 'exp_paper.tar.gz'.
You may just rename or remove the current 'exp_paper_ecml' folder, and uncompress the archive.
It will create again the folder 'exp_paper_ecml', including the results of our experiments.
Therefore, you may just set make_exp=0 and leave make_plots=1 to generate the corresponding plots.

====================================================

GENERAL INSTRUCTIONS

====================================================
1) Inside the matlab folder, create a folder that will contain the setup files (e.g.:
"setfolder"). 

2) Inside the setup folder, FOR EACH FIGURE that you want to be printed, you have to 
create a folder whose name starts with "exp". 

3) You have to put one or more configuration files inside each "exp" folder, depending
on how many curves you have to trace on the figure. The configuration file can be edited 
from  set_params_example.m. EACH CONFIGURATION FILE MUST START WITH "set" (e.g.: set_params_svm_lin_lambda_0).

4) Open main.m inside the matlab folder. Set "setup_folder" to the name of the setup folder
that contains the exp folders (e.g. setup_folder = "setfolder").

5) Set the flag make_exp = 1 (if you want to run the experiment in python) and make_plots = 1 (if you want Matlab
to show plots).

6) Run it!

7) For each exp folder, you will find the plots inside the "fig" folder contained into 
them (plot.pdf).



