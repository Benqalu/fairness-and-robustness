function plotDET(fp, tp, ptitle, color)
%function plotDET(fp,tp, ptitle, color)
%
%  Plot_DET plots detection performance tradeoff on a DET plot
%  and returns the handle for plotting.
%
%  fp and tp are the vectors of true and false
%  positive rates. fn (1-tp) vs fp will be plotted.
%
%  The usage of plotDET is analogous to the standard matlab
%  plot function.
%
%
% ptitle: plot title
% color: e.g., 'b.-'. Default 'b'.
%

Pmiss = 1-tp;
Pfa = fp;

Npts = max(size(Pmiss));
if Npts ~= max(size(Pfa))
        error ('vector size of tp and fp not equal in call to plotDET');
end

%------------------------------
% plot the DET

if (nargin < 3)
    ptitle = '';
    color = 'b';
end

h=plot(ppndf(Pfa), ppndf(Pmiss), color); hold on;
set(h,'DisplayName',ptitle);


pticks = [0.00001 0.00002 0.00005 0.0001  0.0002   0.0005 ...
          0.001   ... %0.002  
          0.005   0.01    0.02     0.05 ...
          0.1     0.2     0.4     0.6     0.8      0.9 ...
          0.95    0.98    0.99    0.995   0.998    0.999 ...
          0.9995  0.9998  0.9999  0.99995 0.99998  0.99999];

xlabels = [' 0.001' ; ' 0.002' ; ' 0.005' ; ' 0.01 ' ; ' 0.02 ' ; ' 0.05 ' ; ...
           '  0.1 ' ; %'  0.2 ' ;
           ' 0.5  ' ; '  1   ' ; '  2   ' ; '  5   ' ; ...
           '  10  ' ; '  20  ' ; '  40  ' ; '  60  ' ; '  80  ' ; '  90  ' ; ...
           '  95  ' ; '  98  ' ; '  99  ' ; ' 99.5 ' ; ' 99.8 ' ; ' 99.9 ' ; ...
           ' 99.95' ; ' 99.98' ; ' 99.99' ; '99.995' ; '99.998' ; '99.999'];

ylabels = xlabels;

%---------------------------
% Get the min/max values of Pmiss and Pfa to plot

DET_limits = Set_DET_limits;


Pmiss_min = DET_limits(1);
Pmiss_max = DET_limits(2);
Pfa_min   = DET_limits(3);
Pfa_max   = DET_limits(4);

%----------------------------
% Find the subset of tick marks to plot

ntick = max(size(pticks));
for n=ntick:-1:1
	if (Pmiss_min <= pticks(n))
		tmin_miss = n;
	end
	if (Pfa_min <= pticks(n))
		tmin_fa = n;
	end
end

for n=1:ntick
	if (pticks(n) <= Pmiss_max)
		tmax_miss = n;
	end
	if (pticks(n) <= Pfa_max)
		tmax_fa = n;
	end
end

%-----------------------------
% Plot the DET grid

set (gca, 'xlim', ppndf([Pfa_min Pfa_max]));
set (gca, 'xtick', ppndf(pticks(tmin_fa:tmax_fa)));
set (gca, 'xticklabel', xlabels(tmin_fa:tmax_fa,:));
set (gca, 'xgrid', 'on');
xlabel ('FP (in %)');


set (gca, 'ylim', ppndf([Pmiss_min Pmiss_max]));
set (gca, 'ytick', ppndf(pticks(tmin_miss:tmax_miss)));
set (gca, 'yticklabel', ylabels(tmin_miss:tmax_miss,:));
set (gca, 'ygrid', 'on')
ylabel ('FN (in %)')

set (gca, 'box', 'on');
axis('square');
axis(axis);




function DET_limits = Set_DET_limits(Pmiss_min, Pmiss_max, Pfa_min, Pfa_max)
% function Set_DET_limits(Pmiss_min, Pmiss_max, Pfa_min, Pfa_max)
%
%  Set_DET_limits initializes the min.max plotting limits for P_min and P_fa.
%
%  See DET_usage for an example of how to use Set_DET_limits.

%cambia qui i limiti!
Pmiss_min_default = 0.0005+eps;
Pmiss_max_default = 0.99-eps;
Pfa_min_default = 0.0005+eps;
Pfa_max_default = 0.99-eps;


if ~(exist('Pmiss_min')); Pmiss_min = Pmiss_min_default; end;
if ~(exist('Pmiss_max')); Pmiss_max = Pmiss_max_default; end;
if ~(exist('Pfa_min')); Pfa_min = Pfa_min_default; end;
if ~(exist('Pfa_max')); Pfa_max = Pfa_max_default; end;

%-------------------------
% Limit bounds to reasonable values

Pmiss_min = max(Pmiss_min,eps);
Pmiss_max = min(Pmiss_max,1-eps);
if Pmiss_max <= Pmiss_min
	Pmiss_min = eps;
	Pmiss_max = 1-eps;
end

Pfa_min = max(Pfa_min,eps);
Pfa_max = min(Pfa_max,1-eps);
if Pfa_max <= Pfa_min
	Pfa_min = eps;
	Pfa_max = 1-eps;
end

%--------------------------
% Load DET_limits with bounds to use

DET_limits = [Pmiss_min Pmiss_max Pfa_min Pfa_max];

