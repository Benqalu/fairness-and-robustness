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

function decision_plot(x_attack,x1,x2,obj,x,y,g_test,lambda)

contourf(x1,x2,obj,50);
colorbar
shading flat;
axis square
hold on;
plot(x(y==-1,1),x(y==-1,2),'c.','MarkerSize',12);
plot(x(y==-1,1),x(y==-1,2),'kO');
plot(x(y==1,1),x(y==1,2),'m.','MarkerSize',12); hold on;
plot(x(y==1,1),x(y==1,2),'kO');
grid on
contour(x1, x2, g_test, [0 0], 'k');
for i=1:numel(x_attack)
    plot(x_attack{i}(:,1),x_attack{i}(:,2),'k','MarkerSize',12);
    plot(x_attack{i}(end,1),x_attack{i}(end,2),'k.','MarkerSize',12);
end
title(['g(x) - \lambda p(x|yc=-1), \lambda=' num2str(lambda)])
axis normal


