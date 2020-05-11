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

function [f t] = FPatFN(fp,tp,th,at_fn)


x(1) = max(1-tp(1-tp <= at_fn));
y(1) = min(fp(1-tp <= at_fn));
thr(1) = min(th(1-tp <= at_fn));

x(2) = min(1-tp(1-tp >= at_fn));
y(2) = max(fp(1-tp >= at_fn));
thr(2) = max(th(1-tp >= at_fn));

if(x(1)==x(2))
    f=y(1);
    t=thr(1);
    return;
end

if(x(1)==0 || x(1) ==1)
    f=y(2);
    t=thr(2);
    return;
end

if(x(2)==0 || x(2) ==1)
    f=y(1);
    t=thr(1);
    return;
end


f=interp1(x,y,at_fn);
t=interp1(x,thr,at_fn);
