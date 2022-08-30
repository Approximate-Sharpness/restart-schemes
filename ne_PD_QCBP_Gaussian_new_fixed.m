clear
close all
clc

import ne_methods.op_matrix_operator 
import ne_methods.h_extract_re_cell_data
import restart_schemes.fom_primal_dual_cb 
import restart_schemes.re_fixed_consts_new

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

%% fixed alpha, eta

alpha = logspace(0,1.25,6);

t = 2000;
max_total_iters = 4000;

for i=1:length(alpha)
    [~, re_ev_cell, re_ii_cell] = re_fixed_consts_new(...
    pd_algo,pd_cost,f,g,kappa,x0,eps0,t,alpha(i),beta,'eval_fns',eval_fns,'total_iters',max_total_iters);

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

savefig(fullfile(dname,'fixed_alpha'))

%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end
