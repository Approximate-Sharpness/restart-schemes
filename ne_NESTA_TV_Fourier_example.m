clear
close all
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
nlevel = 1e-5;    % noise level
sample_rate = 0.15;
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

% flatten the image
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

alpha = 500;   % scaling sharpness constant
beta = 1.0;  % exponent sharpness exponent
t = 500;      % number of restarts

s = sum(abs(opW(x,0)) ~= 0); % sparsity level
f = @(z) norm(opW(z,0),1)/sqrt(s); % objective function
g = @(z) 0;      % gap function
kappa = 0;       % scalar factor for gap function

% here we project the zero vector onto the constraint set, resulting in z0
lmult = max(0,norm(y,2)/nlevel-1);
z0 = (lmult/((lmult+1)*c_A)).*opA(y,1);
eps0 = N*N;

normx = norm(x,2);
% objective value and relative reconstruction error
eval_fns = {f, @(z) norm(z-x,2)/normx};

nesta_cost = @(delta, eps) ceil(4*sqrt(2)*N*delta/eps);
nesta_algo = @(delta, eps, x_init) fom_nesta(...
    x_init, opA, c_A, y, opW, L_W, nesta_cost(delta,eps), nlevel, eps/(N*N), []);


%% Plotting parameters

x_axis_label = 'total iterations';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha

[result, re_ev_cell, re_ii_cell] = re_radial_search(...
nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'alpha',alpha,'beta',beta,'eval_fns',eval_fns);

[re_ev_values, re_ev_indices] = h_extract_re_cell_data(re_ev_cell, re_ii_cell, length(eval_fns));

figure(1)
semilogy(re_ev_indices, re_ev_values(1,:))

xlabel(x_axis_label)
ylabel('objective value')

savefig(fullfile(dname,'objective_value'))

figure(2)
semilogy(re_ev_indices, re_ev_values(2,:))

xlabel(x_axis_label)
ylabel('relative reconstruction error')

savefig(fullfile(dname,'rel_recon_error'))