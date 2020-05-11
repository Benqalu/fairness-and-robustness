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

function x_attack = load_attack_patterns(filename)

fid = fopen(filename);
content = textscan(fid,'%s','delimiter', '\n','bufsize', 5000000);
fclose(fid);

%now content has one cell for each row
content = content{1};

x_attack = cell(size(content,1),1);

for i=1:numel(content)
    k=1;
    [str remain]=strtok(content{i},',');
    str = str2num(str);
    str = str(2:end); %disregards pattern index
    x_attack{i}(k,:) = str;
    while true
        [str, remain] = strtok(remain, ',');
        if isempty(str)
            break;
        end
        k=k+1;
        x_attack{i}(k,:) = str2num(str);
    end
end