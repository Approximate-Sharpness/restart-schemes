clear
close
clc

import ne_methods.op_matrix_operator 
import ne_methods.h_concatenate_cell_entries
import restart_schemes.fom_primal_dual_cb 
import restart_schemes.re_radial_search

% fix seed for debugging
%rng(0)

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
eta = 1e-6;    % sharpness additive
t = 128000;    % num of restart iterations

f = @(z) norm(z,1); % objective function
g = @(z) feasibility_gap(A*z, y, nlevel); % gap function
kappa = 1e1; % scalar factor for gap function

x0 = zeros(N,1);
y0 = zeros(m,1);
eps0 = f(x0) + kappa.*g(x0);

pd_cost = @(delta, eps) ceil(2*L_A*delta/eps);
pd_algo = @(delta, eps, x_init) fom_primal_dual_cb(...
    x_init, y0, delta/L_A, 1/(delta*L_A), pd_cost(delta,eps), opA, y, nlevel, 1);

%% Run restart scheme and standard algorithm to compare

opt_value = f(x) + kappa.*g(x);

[~, o_re_cell, c_re_cell] = re_radial_search(...
    pd_algo,pd_cost,f,g,kappa,x0,eps0,t,'beta',beta);

o_values_re = h_concatenate_cell_entries(o_re_cell);
c_values_re = h_concatenate_cell_entries(c_re_cell);

re_values = o_values_re + kappa.*c_values_re;

[~, o_pd_cell, c_pd_cell] = fom_primal_dual_cb(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), length(re_values), opA, y, nlevel, 1);

o_values_pd = cell2mat(o_pd_cell);
c_values_pd = cell2mat(c_pd_cell);

pd_values = o_values_pd + kappa.*c_values_pd;

%% Generate plots

% NOTE:
% I use `cummin` here since the iterates produced by primal-dual in a
% restart are usually nonfeasible and are 'far away' from minimizing
% f + kappa*g. If you remove the `cummin`, this will become apparent,
% but the final iterates of each restart produce the lowest error.
% It's easier to see the performance if one uses `cummin`.
semilogy([1:length(re_values)], cummin(re_values)-opt_value);
hold on
semilogy([1:length(pd_values)], pd_values-opt_value);
legend({'restarts','no restarts'});
hold off

%% Feasibility gap function handle

function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end