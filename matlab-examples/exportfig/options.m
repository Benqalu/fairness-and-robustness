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


clear opts

opts.format = 'eps';
opts.preview = 'none';
opts.width = w;
opts.height = h;
opts.color = 'rgb';
%opts.defaultfontsize=1;
opts.fontsize = 16;
opts.fontmode='fixed';
%opts.fontmin = 8;
%opts.fontmax = 60;
%opts.defaultlinewidth = 1.1;
opts.linewidth = 1.1;
opts.linemode='fixed';
%opts.linemin = 1.5;
%opts.linemax = 100;
opts.fontencoding = 'latin1';
opts.renderer = 'painters';
opts.resolution = 300;
%opts.stylemap = [];
%opts.applystyle = 0;
%opts.refobj = -1;
opts.bounds = 'tight';
opts.boundscode = 'internal';
explicitbounds = 0;
opts.lockaxes = 0;
opts.separatetext = 0;