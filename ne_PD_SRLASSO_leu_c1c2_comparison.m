clear
close all
clc

% Comparison of selecting grid search parameters c1 and c2. A plot is
% generated for each choice of c1 with varying c2.

import ne_methods.op_matrix_operator 
import restart_schemes.fom_pd_SRLASSO
import restart_schemes.re_radial_pd

% fix seed for debugging
rng(1)

%% SR-LASSO problem definition

data = load('data/leu.mat');
A = data.('features');
b = data.('labels');
lambda = 1;

m = size(A,1);
N = size(A,2);
fprintf('Dimensions of A: %d x %d\n', m, N)
fprintf('Length of b: %d\n', length(b))

% normalize A by standard score
A = (A-mean(A,1))./std(A,0,1);
A = [A ones(m,1)];
opA = @(z,ad) op_matrix_operator(A,z,ad);
L_A = norm(A,2);

%% Restart scheme parameters

x0 = zeros(N+1,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};

f = @(z) lambda*norm(z{1},1) + norm(opA(z{1},0)-b,2);
g = @(z) 0;
kappa = 0;
alpha0 = 1;

eps0 = f({x0});

eval_fns = {f};

pd_cost = @(delta, eps, xy_init) ceil(2*L_A*delta/eps);
pd_algo = @(delta, eps, xy_init, F) fom_pd_SRLASSO(...
    xy_init{1}, xy_init{2}, delta/L_A, 1/(delta*L_A), pd_cost(delta,eps), opA, b, lambda, eval_fns, F);

%% Precompute optimal value with CVX

cvx_precision best
cvx_begin quiet
    variable x(N+1)
    minimize( lambda*norm(x,1) + norm(A*x-b,2) )
cvx_end

opt_value = cvx_optval;

%% Plotting parameters

% x_axis_label = 'total iterations';
% y_axis_label = 'reconstruction error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);


%% Generate plots comparing performance of selecting c1 and c2

c1 = linspace(1,2,6);%[2,4,6,8,10,12];
c2 = linspace(1,2,6);%[2,4,6,8,10,12];

CMAP = linspecer(length(c2));

t = 50000;
max_total_iters = 2500;
ylim_low = 1;

for i=1:length(c1)
    figure
    for j=1:length(c2)
        [~, GRID_VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'a',exp(c1(i)),'alpha0',alpha0,'c1',c1(i),'c2',c2(j),'total_iters',max_total_iters);
        GRID_VALS = modify_values_for_log_plot(GRID_VALS, opt_value);
        semilogy(GRID_VALS,'linewidth',2,'color',CMAP(j,:));
        hold on

        minval = min(GRID_VALS);
        if minval < ylim_low
            ylim_low = minval;
        end
    end
    legend_labels = cell(length(c2),1);
    for j=1:length(c2)
        legend_labels{j} = strcat('$c_2 = $',sprintf(' %1.0f', c2(j)));
    end
    legend(legend_labels,'interpreter','latex','fontsize',14)
    ax=gca; ax.FontSize=14;
    xlim([0,max_total_iters]);  ylim([ylim_low/4,10])
    hold off
    %savefig(fullfile(dname,sprintf('c1c2_comparison_c1_%d',c1(i))))
end


%% Additional functions specific to the experiment
function new_values = modify_values_for_log_plot(values, opt_val)
    new_values = values-opt_val;
    min_nz_val = min(new_values(new_values>0),[],'all');
    new_values = max(new_values,min_nz_val);
end
