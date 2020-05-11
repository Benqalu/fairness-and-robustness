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

function scores = load_attack_patterns(filename)
%This function loads all attack patterns

fid = fopen(filename);
content = textscan(fid,'%s','delimiter', '\n','bufsize', 500000);
fclose(fid);


content = content{1};

scores = cell(size(content,1),1);
k_max = 0;
for i=1:numel(content)
    k=1;
    [str remain]=strtok(content{i},' '); 
    while true
        [str, remain] = strtok(remain, ' ');
        if isempty(str)
            break;
        end
        scores{i}(k) = str2num(str);
        if(k>k_max)
            k_max=k;
        end
        k=k+1;
    end
end

%Create a matrix instead of a cell
scores_ = nan*ones(size(scores,1),k_max);
for i=1:size(scores_,1)
    scores_(i,1:numel(scores{i})) = scores{i};
    
    %Replace nan with the last (minimum) value attained by the attack
    scores_(i,numel(scores{i})+1:end)=scores{i}(end);
end

scores = scores_;


