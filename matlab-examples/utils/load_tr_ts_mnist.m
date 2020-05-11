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

function [x_tr y_tr x_ts y_ts] = load_tr_ts_mnist(folder,pos_class,neg_class)

D = load([folder '/mnist_all.mat']);

pos_tr = eval(['D.train' pos_class]);
neg_tr = eval(['D.train' neg_class]);

pos_ts = eval(['D.test' pos_class]);
neg_ts = eval(['D.test' neg_class]);

x_tr = double([pos_tr; neg_tr]);
y_tr = double([ones(size(pos_tr,1),1); -ones(size(neg_tr,1),1)]);

x_ts = double([pos_ts; neg_ts]);
y_ts = double([ones(size(pos_ts,1),1); -ones(size(neg_ts,1),1)]);

return;
