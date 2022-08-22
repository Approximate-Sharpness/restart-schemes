clear
close
clc

import ne_methods.*
import restart_schemes.fom_nesta
import restart_schemes.re_radial_search

% fix seed for debugging
%rng(0)

%% TV minimization problem definition

% load image
X = double(imread('data/GPLU_phantom_512.png'))/255;

[N, ~] = size(X); % image size (assumed to be N by N)
nlevel = 1e-6;    % noise level
sample_rate = 0.12;
m = ceil(sample_rate*N*N);

% generate sampling mask
var_hist = sa_inverse_square_law_hist_2d(N,1);
var_probs = sa_bernoulli_sampling_probs_2d(var_hist,m/2);
var_mask = binornd(ones(N,N),var_probs);

num_var_samples = sum(var_mask,'all');
uni_mask_cond = rand(N*N-num_var_samples,1) <= (m/2)/(N*N-num_var_samples);

uni_mask = zeros(N,N);
uni_mask(~var_mask) = uni_mask_cond;

% logical OR the two masks
mask = uni_mask | var_mask;
sample_idxs = find(mask);
m_exact = length(sample_idxs);

%imshow(mask)
%imshow(uni_mask)
%imshow(var_mask)

% flat image
x = reshape(X,[N*N,1]);

% measurement matrix (subsampled Fourier)
opA = @(z,ad) (N/sqrt(m))*op_fourier_2d(z, ad, N, sample_idxs);
c_A = N*N/m;

% measurement vector
e = randn(m_exact,1) + 1i*randn(m_exact,1);
y = opA(x,0) + nlevel*e/norm(e);

% analysis matrix (anisotropic TV matrix)
opW = @(z,ad) op_discrete_gradient_2d(z,ad,N,N);
L_W = 2*sqrt(2);

%% Restart scheme parameters

beta = 1.0;    % sharpness exponent
t = 2000;        % num of restart iterations

s = sum(abs(opW(x,0)) ~= 0); % sparsity level
f = @(z) norm(opW(x,0),1)/sqrt(s); % objective function
g = @(z) 0;      % gap function
kappa = 0;       % scalar factor for gap function

% here we project the zero vector onto the constraint set, resulting in z0
lmult = max(0,norm(y,2)/nlevel-1);
z0 = (lmult/((lmult+1)*c_A)).*opA(y,1);
eps0 = f(z0) + kappa.*g(z0);

nesta_cost = @(delta, eps) ceil(2*N*delta/eps);
nesta_algo = @(delta, eps, x_init) fom_nesta(...
    x_init, opA, c_A, y, opW, L_W, nesta_cost(delta,eps), nlevel, eps/(N*N), 1);

%% Run restart scheme and standard algorithm to compare

opt_value = f(x) + kappa.*g(x);

[~, o_re_cell, c_re_cell] = re_radial_search(...
    nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'beta',beta);

o_values_re = h_concatenate_cell_entries(o_re_cell);
c_values_re = h_concatenate_cell_entries(c_re_cell);

re_values = o_values_re/sqrt(s) + kappa.*c_values_re;

[~, o_nesta_cell, c_nesta_cell] = fom_nesta(...
    z0, opA, c_A, y, opW, L_W, length(re_values), nlevel, 1e-4, 1);

o_values_nesta = cell2mat(o_nesta_cell);
c_values_nesta = cell2mat(c_nesta_cell);

nesta_values = o_values_nesta/sqrt(s) + kappa.*c_values_nesta;

%% Generate plots

semilogy([1:length(re_values)], cummin(re_values));
hold on
semilogy([1:length(nesta_values)], nesta_values);
legend({'restarts','no restarts'});
hold off

%imshow(reshape(abs(x_rec),[N,N]));
