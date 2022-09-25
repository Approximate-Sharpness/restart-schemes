clear
close all
clc

import ne_methods.op_matrix_operator 
import restart_schemes.fom_pd_SRLASSO
import restart_schemes.re_radial_pd

% fix seed for debugging
rng(1)

%% SR-LASSO problem definition

data = load('data/winequality.mat');
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

% precompute optimal value with CVX
cvx_precision best
cvx_begin quiet
    variable x(N+1)
    minimize( lambda*norm(x,1) + norm(A*x-b,2) )
cvx_end

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

%% Plotting parameters

% x_axis_label = 'total iterations';
% y_axis_label = 'reconstruction error';
ylim_low = 1e-10;

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha and fixed beta
beta = 1;
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 4000;
max_total_iters = 2000;

figure
for i=1:length(alpha)
    [~, VALS] = re_radial_pd(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'r',exp(-1),'a',exp(beta),'beta',beta,'alpha',alpha(i),'total_iters',max_total_iters);
    semilogy(VALS-cvx_optval,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([ylim_low,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% fixed alpha and search over beta
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 100000;
max_total_iters = 3000;

figure
for i=1:length(alpha)
    [~, VALS] = re_radial_pd(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'r',exp(-1),'alpha',alpha(i),'total_iters',max_total_iters);
    semilogy(VALS-cvx_optval,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([ylim_low,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_search_beta'))

clear -regexp ^VALS;
clear legend_labels;


%% fixed beta and search over alpha
beta = 1:0.5:3;
CMAP = linspecer(length(beta));

t = 10000;
max_total_iters = 3000;

figure
for i=1:length(beta)
    [~, VALS] = re_radial_pd(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'r',exp(-1),'a',exp(beta(i)),'beta',beta(i),'total_iters',max_total_iters);
    semilogy(VALS-cvx_optval,'linewidth',2','color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(beta),1);
for i=1:length(beta)
    legend_labels{i} = strcat('$\beta = $',sprintf(' %1.1f', beta(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([ylim_low,max(VALS)])
hold off
savefig(fullfile(dname,'search_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% compare standard PD with radial-grid restart scheme

beta2 = 1;
alpha1 = 1e1;
beta1 = 1;

t = 12000;
max_total_iters = 3000;

figure

[~, pd_ev_values] = fom_pd_SRLASSO(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), max_total_iters, opA, b, lambda, eval_fns, f);

for i=1:3
    if i == 1
        [xfin, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha',alpha1,'a',exp(beta1),'beta',beta1,'total_iters',max_total_iters);
        opt_value = f(xfin) + kappa*g(xfin);
    elseif i == 2
        [~, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'a',exp(beta2),'beta',beta2,'total_iters',max_total_iters);
    elseif i == 3
        [~, VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'total_iters',max_total_iters);
    end
    semilogy(VALS-cvx_optval,'linewidth',2);
    hold on
end

semilogy(pd_ev_values-cvx_optval,'linewidth',2);

legend_labels = cell(3,1);
legend_labels{1} = strcat('$\alpha = $',sprintf(' %1.1f,', alpha1),' $\beta = $',sprintf(' %1.1f', beta1));
legend_labels{2} = strcat('$\alpha$-grid,',' $\beta = $',sprintf(' %1.1f', beta2));
legend_labels{3} = '$(\alpha,\beta)$-grid';
legend_labels{4} = 'no restarts';
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([ylim_low,max(pd_ev_values)])

hold off

savefig(fullfile(dname,'tweaks_pd_comparison'))

clear -regexp ^VALS;
