clear
close all
clc

import ne_methods.op_matrix_operator 
import ne_methods.h_extract_re_cell_data
import restart_schemes.fom_primal_dual_cb 
import restart_schemes.re_radial_search

% fix seed for debugging
rng(1729)

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
L_A = sqrt(eigs(A'*A,1)); % Lipschitz constant

% measurement vector
e = randn(m,1);
y = A*x + nlevel*e/norm(e);

%% Restart scheme parameters

beta = 1.0;    % sharpness exponent

% these are defined per subexperiment; do not uncomment these!
% t = 10000;    % num of restart iterations
% max_total_iters = 2000; % maximum number of total iterations to run

f = @(z) norm(z,1); % objective function
g = @(z) feasibility_gap(A*z, y, nlevel); % gap function
kappa = 1e1; % scalar factor for gap function

x0 = zeros(N,1);
y0 = zeros(m,1);
opt_value = f(x) + kappa.*g(x);
eps0 = f(x0) + kappa.*g(x0);

eval_fns = {f, g};

pd_cost = @(delta, eps) ceil(2*L_A*kappa*delta/eps);
pd_algo = @(delta, eps, x_init) fom_primal_dual_cb(...
    x_init, y0, delta/(kappa*L_A), kappa/(delta*L_A), pd_cost(delta,eps), opA, y, nlevel, []);


%% Plotting parameters

x_axis_label = 'total iterations';
y_axis_label = 'objective-gap sum error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% grid search over alpha, eta

t = 150000;
max_total_iters = 100000;

[~, re_ev_cell, re_ii_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'eval_fns',eval_fns,'total_iters',max_total_iters);

[re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

re_values = re_ev_values(1,:) + kappa.*re_ev_values(2,:);

figure(1)

semilogy(re_ev_indices, re_values-opt_value);
xlabel(x_axis_label)
ylabel(y_axis_label)
savefig(fullfile(dname,'grid_search_alpha_eta'))

clear -regexp ^re_;

%% grid search over alpha, fixed eta

eta = nlevel*[1e-1 1e-0 1e1 1e2 1e3];

t = 50000;
max_total_iters = 20000;

exp2_best_eta = 0; % track the best eta value
exp2_best_eta_value = Inf;

figure(2)

for i=1:length(eta)
    [~, re_ev_cell, re_ii_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'eta',eta(i),'eval_fns',eval_fns,'total_iters',max_total_iters);

    [re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

    re_values = re_ev_values(1,:) + kappa.*re_ev_values(2,:);
    
    min_re_value = min(re_values);
    if min_re_value < exp2_best_eta_value
        exp2_best_eta = eta(i);
        exp2_best_eta_value = min_re_value;
    end
    
    semilogy(re_ev_indices, abs(re_values-opt_value))
    
    hold on
end

xlabel(x_axis_label)
ylabel(y_axis_label)

legend_labels = cell(length(eta),1);
for i=1:length(eta)
    legend_labels{i} = sprintf('\\eta = 10^{%d}', log10(eta(i)));
end

legend(legend_labels)

hold off

savefig(fullfile(dname,'grid_search_alpha-fixed_eta'))

clear -regexp ^re_;
clear legend_labels;

%% grid search over eta, fixed alpha

alpha = [0.1 0.5 1 2 5];

t = 50000;
max_total_iters = 20000;

exp3_best_alpha = 0; % track the best eta value
exp3_best_alpha_value = Inf;

figure(3)

for i=1:length(alpha)
    [~, re_ev_cell, re_ii_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'alpha',alpha(i),'eval_fns',eval_fns,'total_iters',max_total_iters);

    [re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

    re_values = re_ev_values(1,:) + kappa.*re_ev_values(2,:);
    
    min_re_value = min(re_values);
    if min_re_value < exp3_best_alpha_value
        exp3_best_alpha = alpha(i);
        exp3_best_alpha_value = min_re_value;
    end
    
    semilogy(re_ev_indices, abs(re_values-opt_value))
    
    hold on
end

xlabel(x_axis_label)
ylabel(y_axis_label)

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = sprintf('\\alpha = %s', num2str(alpha(i)));
end

legend(legend_labels)

hold off

savefig(fullfile(dname,'grid_search_eta-fixed_alpha'))

clear -regexp ^re_;
clear legend_labels;

%% fixed alpha, eta

eta = 1e-1*nlevel;
alpha = logspace(-1,0.25,6);

t = 3000;
max_total_iters = 3000;

figure(4)

for i=1:length(alpha)
    [~, re_ev_cell, re_ii_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'alpha',alpha(i),'eta',eta,'eval_fns',eval_fns,'total_iters',max_total_iters);

    [re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

    re_values = re_ev_values(1,:) + kappa.*re_ev_values(2,:);

    semilogy(re_ev_indices, abs(re_values-opt_value))

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

savefig(fullfile(dname,'fixed_alpha_eta'))

clear -regexp ^re_;
clear legend_labels;

%% compare standard PD with radial-grid restart scheme

t = 150000;
max_total_iters = 50000;

figure(5)

[~, pd_ev_values] = fom_primal_dual_cb(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), max_total_iters, opA, y, nlevel, eval_fns);

pd_values = pd_ev_values(1,:) + kappa.*pd_ev_values(2,:);

for i=1:3
    if i == 1
        [~, re_ev_cell, re_ii_cell] = re_radial_search(...
            pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'eval_fns',eval_fns,'total_iters',max_total_iters);
    elseif i == 2
        [~, re_ev_cell, re_ii_cell] = re_radial_search(...
            pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'eta',exp2_best_eta,'eval_fns',eval_fns,'total_iters',max_total_iters);
    elseif i == 3
        [~, re_ev_cell, re_ii_cell] = re_radial_search(...
            pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta,'alpha',exp3_best_alpha,'eval_fns',eval_fns,'total_iters',max_total_iters);
    end

    [re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

    re_values = re_ev_values(1,:) + kappa.*re_ev_values(2,:);
    
    semilogy(re_ev_indices, abs(re_values-opt_value))
    
    hold on
end

semilogy([1:max_total_iters], abs(pd_values-opt_value));

xlabel(x_axis_label)
ylabel(y_axis_label)

legend_labels = cell(4,1);
legend_labels{1} = '\alpha,\eta-grid';
legend_labels{2} = sprintf('\\alpha-grid, \\eta = 10^{%s}', num2str(log10(exp2_best_eta)));
legend_labels{3} = sprintf('\\eta-grid, \\alpha = %s', num2str(exp3_best_alpha));
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
