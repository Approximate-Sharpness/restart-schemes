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
L_A = norm(A,2); % Lipschitz constant

% measurement vector
e = randn(m,1);
b = A*x + nlevel*e/norm(e);

%% Restart scheme parameters

alpha = 10.0; % sharpness scaling
beta = 1.0;   % sharpness exponent

t = 1000;   % num of restart iterations
max_total_iters = 500; % maximum number of total iterations

f = @(z) norm(z{1},1); % objective function
g = @(z) feasibility_gap(A*z{1}, b, nlevel); % gap function
kappa = 1e1; % scalar factor for gap function

x0 = zeros(N,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};
opt_value = f({x}) + kappa.*g({x});
eps0 = f({x0}) + kappa.*g({x0});

eval_fns = {f, g};

pd_cost = @(delta, eps) ceil(2*L_A*kappa*delta/eps);
pd_algo = @(delta, eps, xy_init) fom_primal_dual_cb(...
    xy_init{1}, y0, delta/(kappa*L_A), kappa/(delta*L_A), pd_cost(delta,eps), opA, b, nlevel, []);


%% Plotting parameters

x_axis_label = 'total iterations';
y_axis_label = 'objective-gap-sum error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha, eta

[~, re_ev_cell, re_ii_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha',alpha,'beta',beta,'eval_fns',eval_fns,'total_iters',max_total_iters);

[re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

re_values = re_ev_values(1,:) + kappa.*re_ev_values(2,:);

semilogy(re_ev_indices, abs(re_values-opt_value))

xlabel(x_axis_label)
ylabel(y_axis_label)

savefig(fullfile(dname,'objective_value'))

%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end
