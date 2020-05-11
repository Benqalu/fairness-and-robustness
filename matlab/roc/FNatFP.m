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

function [f t] = FNatFP(fp,tp,th,at_fp)

if(sum(fp <= at_fp)==0)
    f=max(1-tp(fp >= at_fp));
    t=max(th(fp >= at_fp));
    return;
end

if(sum(fp >= at_fp)==0)
    f=min(1-tp(fp <= at_fp));
    t=min(th(fp <= at_fp));
    return;
end

x(1) = max(fp(fp <= at_fp));
y(1) = min(1-tp(fp <= at_fp));
thr(1) = min(th(fp <= at_fp));

x(2) = min(fp(fp >= at_fp));
y(2) = max(1-tp(fp >= at_fp));
thr(2) = max(th(fp >= at_fp));

if(x(1)==x(2))
    f=y(1);
    t = thr(1);
    return;
end



f=interp1(x,y,at_fp);
t=interp1(x,thr,at_fp);

return;