clear
close
clc

import ne_methods.op_matrix_operator 
import ne_methods.h_concatenate_cell_entries
import restart_schemes.fom_fista 
import restart_schemes.re_radial_search

% fix seed for debugging
%rng(0)

%% LASSO problem definition

N = 128;          % s-sparse vector size ground truth vector size
s = 10;           % sparsity level
m = 32;           % measurements
sigma = 1e-6;     % noise level
lambda = sigma;   % U-LASSO parameter

% s-sparse vector
x = zeros(N,1);
x(1:s) = randn(s,1);
x = x(randperm(N));

% measurement matrix (Gaussian random)
A = randn(m,N)/sqrt(m);
opA = @(z,ad) op_matrix_operator(A,z,ad);
L_A = eigs(A'*A,1); % Lipschitz constant
stepsize = 1/L_A;   % FISTA step size

% measurement vector
e = randn(m,1);
y = A*x + sigma*e/norm(e);

%% Restart scheme parameters

beta = 1;      % sharpness exponent
eta = sigma;   % sharpness additive
t = 100;     % num of restart iterations

f = @(z) lambda*norm(z,1) + 0.5*norm(A*z-y,2)^2;
g = @(z) 0;
kappa = 0;

x0 = zeros(N,1);
eps0 = f(x0);

fista_cost = @(delta, eps) ceil(sqrt(2*L_A*(delta^2)/eps));
fista_algo = @(delta, eps, x_init) fom_fista(...
    x_init, opA, y, lambda, stepsize, fista_cost(delta,eps), 1);


%% Run restart scheme and standard algorithm to compare

opt_value = f(x);

[~, o_re_cell, c_re_cell] = re_radial_search(...
    fista_algo,fista_cost,f,g,kappa,x0,eps0,t,'beta',beta,'eta',eta);

o_values_re = h_concatenate_cell_entries(o_re_cell);
c_values_re = h_concatenate_cell_entries(c_re_cell);

% NOTE:
% `re_values` are expressed like this due to the output of fom_fista.m
re_values = lambda.*o_values_re + 0.5.*(c_values_re.^2);

[~, o_fista_cell, c_fista_cell] = fom_fista(...
    x0, opA, y, lambda, stepsize, length(re_values), 1);
    
o_values_fista = cell2mat(o_fista_cell);
c_values_fista = cell2mat(c_fista_cell);

% NOTE:
% `fista_values` are expressed like this due to the output of fom_fista.m
fista_values = lambda.*o_values_fista + 0.5.*(c_values_fista.^2);

%% Generate plots

semilogy([1:length(re_values)], cummin(re_values)-opt_value);
hold on
semilogy([1:length(fista_values)], fista_values-opt_value);
legend({'restarts','no restarts'});
hold off