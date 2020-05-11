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

function [fp tp th] = ROCapprox(n,p)
%Approximated ROC, faster (1000 points),
%it does not require one to build the mex file roc_foc.c

if(size(n,1)>1)
    n=n';
end
if(size(p,1)>1)
    p=p';
end

%Quantization step
N=999;
%Normalize scores in [0,1]
maximum = max([n p]);
minimum = min([n p]);
n = (n-minimum)/(maximum-minimum);
p = (p-minimum)/(maximum-minimum);
fp=zeros(1,N+1);
tp=zeros(1,N+1);
for th=0:N
    fp(th+1)=sum(n>th/N);
    tp(th+1)=sum(p>th/N);
end

%Rebuld ROC thresholds
th=0:N;
th=th/N*(maximum-minimum)+minimum;

tp=tp/numel(p);
fp=fp/numel(n);