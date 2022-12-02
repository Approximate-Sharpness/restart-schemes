clear
close all
clc

% Performance of various restart schemes with different approximate 
% sharpness parameters on a sparse recovery problem.

import ne_methods.op_matrix_operator 
import restart_schemes.fom_pd_SRLASSO
import restart_schemes.re_radial_pd

% fix seed for debugging
rng(1)

%% SR-LASSO problem definition

data = load('data/colon-cancer.mat');
A = data.('features');
b = data.('labels');

lambda = 2;

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
beta0 = 5;
c1 = 4;
c2 = 4;

eps0 = f({x0});

eval_fns = {f};

pd_cost = @(delta, eps, xy_init) ceil(2*L_A*delta/eps);
pd_algo = @(delta, eps, xy_init, F) fom_pd_SRLASSO(...
    xy_init{1}, xy_init{2}, delta/L_A, 1/(delta*L_A), pd_cost(delta,eps), opA, b, lambda, eval_fns, F);

scheme = @(t,varargin) re_radial_pd(pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha0',alpha0,'beta0',beta0,'c1',c1,'c2',c2,varargin{:});

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

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha and fixed beta
beta = beta0;
alpha = logspace(0,2,11);
CMAP = linspecer(length(alpha));

t = 10000;
total_iters = 5000;
ylim_low = 1;

figure
for i=1:length(alpha)
    [~, VALS] = scheme(t,'alpha',alpha(i),'beta',beta,'total_iters',total_iters);
    VALS = modify_values_for_log_plot(VALS,opt_value);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
    
    minval = min(VALS);
    if minval < ylim_low
        ylim_low = minval;
    end
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([ylim_low/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% fixed alpha and search over beta
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 15000;
total_iters = 5000;
ylim_low = 1;

figure
for i=1:length(alpha)
    [~, VALS] = scheme(t,'alpha',alpha(i),'total_iters',total_iters);
    VALS = modify_values_for_log_plot(VALS,opt_value);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
    
    minval = min(VALS);
    if minval < ylim_low
        ylim_low = minval;
    end
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([ylim_low/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_search_beta'))

clear -regexp ^VALS;
clear legend_labels;


%% fixed beta and search over alpha
beta = 1:0.5:6;
CMAP = linspecer(length(beta));

t = 20000;
total_iters = 10000;
ylim_low = 1;

figure
for i=1:length(beta)
    [~, VALS] = scheme(t,'a',exp(c1*beta(i)),'beta',beta(i),'total_iters',total_iters);
    VALS = modify_values_for_log_plot(VALS,opt_value);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
    
    minval = min(VALS);
    if minval < ylim_low
        ylim_low = minval;
    end
end

legend_labels = cell(length(beta),1);
for i=1:length(beta)
    legend_labels{i} = strcat('$\beta = $',sprintf(' %1.1f', beta(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([ylim_low/4,max(VALS)])
hold off
savefig(fullfile(dname,'search_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% compare standard PD with radial-grid restart scheme

alpha3 = alpha0;
beta2 = beta0;
alpha1 = alpha0;
beta1 = beta0;

t = 40000;
max_total_iters = 20000;
ylim_low = 1;

figure

[~, pd_ev_values] = fom_pd_SRLASSO(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), max_total_iters, opA, b, lambda, eval_fns, f);

for i=1:4
    if i == 1
        [~, VALS] = scheme(t,'alpha',alpha1,'beta',beta1,'total_iters',max_total_iters);
    elseif i == 2
        [~, VALS] = scheme(t,'a',exp(c1*beta2),'beta',beta2,'total_iters',max_total_iters);
    elseif i == 3
        [~, VALS] = scheme(t,'alpha',alpha3,'total_iters',max_total_iters);
    elseif i == 4
        [~, VALS] = scheme(t,'a',exp(c1),'total_iters',max_total_iters);
    end
    VALS = modify_values_for_log_plot(VALS,opt_value);
    semilogy(VALS,'linewidth',2);
    hold on
    
    minval = min(VALS);
    if minval < ylim_low
        ylim_low = minval;
    end
end

semilogy(modify_values_for_log_plot(pd_ev_values,opt_value),'linewidth',2,'linestyle','--');

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
