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

function p=pKDE(x,y,x_test,distance,gamma)

neg_samples = x(y==-1,:);
p=zeros(1,size(x_test,1));
for i=1:size(x_test,1)
    dist = repmat(x_test(i,:),size(neg_samples,1),1) - neg_samples; %distanza tra x_test i-esimo e tutti i negative
    dist_scalar = zeros(1,size(dist,1));
    for j=1:size(dist,1)
        if(strcmp(distance,'kde_euclidean'))
            dist_scalar(j) = exp(-gamma*norm(dist(j,:))^2); %EUCLIDEAN
        else
            dist_scalar(j) = exp(-gamma*sum(abs(dist(j,:)))); %HAMMING
        end
        
    end
   
    p(i) = gamma*sum(dist_scalar)/numel(dist_scalar);
end
