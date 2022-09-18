clear
close all
clc

import ne_methods.op_matrix_operator 
import ne_methods.h_extract_re_cell_data
import restart_schemes.fom_pd_QCBP 
import restart_schemes.re_radial_search

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
kappa = 1e1; % scalar factor for gap function

x0 = zeros(N,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};
opt_value = f({x}) + kappa.*g({x});
eps0 = f(x0y0) + kappa.*g(x0y0);

eval_fns = {@(z) norm(z{1}-x,2)};

pd_cost = @(delta, eps) ceil(2*L_A*kappa*delta/eps);
pd_algo = @(delta, eps, xy_init) fom_pd_QCBP(...
    xy_init{1}, y0, delta/(kappa*L_A), kappa/(delta*L_A), pd_cost(delta,eps), opA, b, nlevel, []);


%% Plotting parameters

x_axis_label = 'total iterations';
y_axis_label = 'reconstruction error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha, beta

beta = 1;
alpha = logspace(0.25,1.25,5);

t = 10000;
max_total_iters = 2500;

figure(1)

for i=1:length(alpha)
    [~, re_ev_cell, re_ii_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'beta',beta,'alpha',alpha(i),'eval_fns',eval_fns,'total_iters',max_total_iters);

    [re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

    semilogy(re_ev_indices, re_ev_values)

    hold on
end

xlabel(x_axis_label)
ylabel(y_axis_label)

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = sprintf('log_{10}(\\alpha) = %s', num2str(log10(alpha(i))));
end

legend(legend_labels)

hold off

savefig(fullfile(dname,'fixed_alpha_beta'))

clear -regexp ^re_;
clear legend_labels;

%% compare standard PD with radial-grid restart scheme

beta2 = 2;
alpha1 = 1e1;
beta1 = 1;

t = 100000;
max_total_iters = 30000;

figure(2)

[~, pd_ev_values] = fom_pd_QCBP(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), max_total_iters, opA, b, nlevel, eval_fns);

for i=1:3
    if i == 1
        [xfin, re_ev_cell, re_ii_cell] = re_radial_search(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha',alpha1,'beta',beta1,'eval_fns',eval_fns,'total_iters',max_total_iters);
        opt_value = f(xfin) + kappa*g(xfin);
    elseif i == 2
        [~, re_ev_cell, re_ii_cell] = re_radial_search(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'beta',beta2,'eval_fns',eval_fns,'total_iters',max_total_iters);
    elseif i == 3
        [~, re_ev_cell, re_ii_cell] = re_radial_search(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'eval_fns',eval_fns,'total_iters',max_total_iters);
    end

    [re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));
    
    semilogy(re_ev_indices, re_ev_values)
    
    hold on
end

semilogy([1:max_total_iters], pd_ev_values);

xlabel(x_axis_label)
ylabel(y_axis_label)

legend_labels = cell(3,1);
legend_labels{1} = sprintf('\\alpha = %s, \\beta = %s', num2str(alpha1), num2str(beta1));
legend_labels{2} = sprintf('\\alpha-grid, \\beta = %s', num2str(beta2));
legend_labels{3} = '(\alpha,\beta)-grid';
legend_labels{4} = 'no restarts';
legend(legend_labels)

hold off

savefig(fullfile(dname,'standard_pd_comparison'))

clear -regexp ^re_;



%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end
