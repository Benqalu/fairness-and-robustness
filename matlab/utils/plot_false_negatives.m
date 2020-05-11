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

function fig = plot_false_negatives(fig,classifier_params,constraint_params, mimicry_params, dMAX,falneg_perfect, falneg_lim,plot_params)

%This function plots false negatives

%Define plots parameters
xlabel('d_{max}');
ylabel('FN');
hold on
title(plot_params.title)


%PLOT PERFECT KNOWLEDGE (PK)
if(constraint_params.max_boundaries > 1)
    h = errorbar(dMAX,mean(falneg_perfect,1),0.5*std(falneg_perfect,[],1),plot_params.color );
    axis([0,dMAX(end),0,1]);
else
    h = errorbar(0:numel(mean(falneg_perfect,1))-1,mean(falneg_perfect,1),0.5*std(falneg_perfect,[],1),plot_params.color);
    axis([0,numel(mean(falneg_perfect,1))-1,0,1]);
end


set(h,'DisplayName', plot_params.legend_pk)
legend location Southeast
legend show
grid on

%PLOT LIMITED KNOWLEDGE (LK)
title(plot_params.title)
if(constraint_params.max_boundaries > 1)
    h = errorbar(dMAX,mean(falneg_lim,1),0.5*std(falneg_lim,[],1),[plot_params.color '--']);
    axis([0,dMAX(end),0,1]);
else
    h = errorbar(0:numel(mean(falneg_lim,1))-1,mean(falneg_lim,1),0.5*std(falneg_lim,[],1),[plot_params.color '--']);
    axis([0,numel(mean(falneg_lim,1))-1,0,1]);
end

set(h,'DisplayName', plot_params.legend_lk)
legend location Southeast
legend show
grid on



