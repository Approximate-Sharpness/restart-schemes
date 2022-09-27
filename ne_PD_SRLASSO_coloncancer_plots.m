clear
close all
clc

import ne_methods.op_matrix_operator 
import restart_schemes.fom_pd_SRLASSO
import restart_schemes.re_radial_pd

% fix seed for debugging
rng(1)

%% SR-LASSO problem definition

data = load('data/colon-cancer.mat');
A = data.('features');
b = data.('labels');
lambda = 2.75;

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

s = length(x(abs(x) > 1e-5));
fprintf('CVX solution has %d coefficients of absolute value > 1e-5\n',s)

%% Plotting parameters

% x_axis_label = 'total iterations';
% y_axis_label = 'reconstruction error';
ylim_low = 1;

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);



%% compare standard PD with radial-grid restart scheme

alpha3 = 1;
beta2 = 1;
alpha1 = 1;
beta1 = 1;

t = 120000;
max_total_iters = 40000;

figure

[~, pd_ev_values] = fom_pd_SRLASSO(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), max_total_iters, opA, b, lambda, eval_fns, f);

for i=1:4
    if i == 1
        [~, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha',alpha1,'a',exp(beta1),'beta',beta1,'total_iters',max_total_iters);
    elseif i == 2
        [~, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'a',exp(beta2),'beta',beta2,'total_iters',max_total_iters);
    elseif i == 3
        [~, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha',alpha3,'total_iters',max_total_iters);
    elseif i == 4
        [~, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'total_iters',max_total_iters);
    end
    VALS = modify_values_for_log_plot(VALS,opt_value);
    semilogy(VALS,'linewidth',2);
    hold on
    
    minval = min(VALS);
    if minval < ylim_low
        ylim_low = minval;
    end
end

semilogy(modify_values_for_log_plot(pd_ev_values,opt_value),'linewidth',2);

legend_labels = cell(5,1);
legend_labels{1} = strcat('$\alpha = $',sprintf(' %1.1f,', alpha1),' $\beta = $',sprintf(' %1.1f', beta1));
legend_labels{2} = strcat('$\alpha$-grid,',' $\beta = $',sprintf(' %1.1f', beta2));
legend_labels{3} = strcat(' $\alpha = $',sprintf(' %1.1f', alpha3),', $\beta$-grid');
legend_labels{4} = '$(\alpha,\beta)$-grid';
legend_labels{5} = 'no restarts';
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([ylim_low/4,max(pd_ev_values)])

hold off

savefig(fullfile(dname,'pd_comparison'))

clear -regexp ^VALS;

function new_values = modify_values_for_log_plot(values, opt_val)
    new_values = values-opt_val;
    min_nz_val = min(new_values(new_values>0),[],'all');
    new_values = max(new_values,min_nz_val);
end
