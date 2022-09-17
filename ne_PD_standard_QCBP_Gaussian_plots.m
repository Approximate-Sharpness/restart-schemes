clear
close all
clc

import ne_methods.op_matrix_operator 
import ne_methods.h_extract_re_cell_data
import restart_schemes.fom_primal_dual_QCBP 
import restart_schemes.re_radial_search2

% fix seed for debugging
rng(1)

%% QCBP problem definition

N = 128;          % s-sparse vector size
s = 10;           % sparsity
m = 60;           % measurements
nlevel = 1e-6;    % noise level

% s-sparse vector
x = zeros(N,1);
x(1:s) = randn(s,1);
x = x(randperm(N));

% measurement matrix (Gaussian random)
A = randn(m,N)/sqrt(m);
opA = @(z,ad) op_matrix_operator(A,z,ad);
L_A = norm(A,2); % Lipschitz constant

% measurement vector
e = randn(m,1);
b = A*x + nlevel*e/norm(e);

%% Restart scheme parameters

f = @(z) norm(z{1},1)/sqrt(s); % objective function
g = @(z) feasibility_gap(A*z{1}, b, nlevel); % gap function
kappa = 10;%1e1; % scalar factor for gap function

x0 = zeros(N,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};
opt_value = f({x}) + kappa.*g({x});
eps0 = f(x0y0) + kappa.*g(x0y0);

eval_fns = {@(z) norm(z{1}-x,2)};

pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
pd_algo = @(delta, eps, xy_init,F) fom_primal_dual_QCBP(...
    xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);


%% Plotting parameters

% x_axis_label = 'total iterations';
% y_axis_label = 'reconstruction error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha and fixed beta
beta = 1;
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 10000;
max_total_iters = 2500;

figure
for i=1:length(alpha)
    [~, ~, ~, VALS] = re_radial_search2(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'r',exp(-1),'a',exp(beta),'beta',beta,'alpha',alpha(i),'eval_fns',eval_fns,'total_iters',max_total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_fixed_beta'))

clear -regexp ^re_;
clear legend_labels;

%% fixed alpha and search over beta
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 10000;
max_total_iters = 2500;

figure
for i=1:length(alpha)
    [~, ~, ~, VALS] = re_radial_search2(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'r',exp(-1),'alpha',alpha(i),'eval_fns',eval_fns,'total_iters',max_total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_search_beta'))

clear -regexp ^re_;
clear legend_labels;


%% fixed beta and search over alpha
beta = 1:0.5:3;

t = 10000;
max_total_iters = 2500;

figure
for i=1:length(beta)
    [~, ~, ~, VALS] = re_radial_search2(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'r',exp(-1),'a',exp(beta(i)),'beta',beta(i),'eval_fns',eval_fns,'total_iters',max_total_iters);
    semilogy(VALS,'linewidth',2);
    hold on
end

legend_labels = cell(length(beta),1);
for i=1:length(beta)
    legend_labels{i} = strcat('$\beta = $',sprintf(' %1.1f', beta(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,max(VALS)])
hold off
savefig(fullfile(dname,'search_alpha_fixed_beta'))

clear -regexp ^re_;
clear legend_labels;

%% compare standard PD with radial-grid restart scheme

beta2 = 1;
alpha1 = 1e1;
beta1 = 1;

t = 100000;
max_total_iters = 2500;

figure

[~, pd_ev_values] = fom_primal_dual_QCBP(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), max_total_iters, opA, b, nlevel, eval_fns,@(x) 0);

for i=1:3
    if i == 1
        [xfin, re_ev_cell, re_ii_cell, VALS] = re_radial_search2(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha',alpha1,'a',exp(beta1),'beta',beta1,'eval_fns',eval_fns,'total_iters',max_total_iters);
        opt_value = f(xfin) + kappa*g(xfin);
    elseif i == 2
        [~, re_ev_cell, re_ii_cell, VALS] = re_radial_search2(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'a',exp(beta2),'beta',beta2,'eval_fns',eval_fns,'total_iters',max_total_iters);
    elseif i == 3
        [~, re_ev_cell, re_ii_cell, VALS] = re_radial_search2(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'eval_fns',eval_fns,'total_iters',max_total_iters);
    end
    semilogy(VALS,'linewidth',2);
    hold on
end

semilogy([1:max_total_iters], pd_ev_values,'linewidth',2);

legend_labels = cell(3,1);
legend_labels{1} = strcat('$\alpha = $',sprintf(' %1.1f,', alpha1),' $\beta = $',sprintf(' %1.1f', beta1));
legend_labels{2} = strcat('$\alpha$-grid,',' $\beta = $',sprintf(' %1.1f', beta2));
legend_labels{3} = '$(\alpha,\beta)$-grid';
legend_labels{4} = 'no restarts';
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,max(pd_ev_values)])

hold off

savefig(fullfile(dname,'standard_pd_comparison'))

clear -regexp ^re_;



%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end
