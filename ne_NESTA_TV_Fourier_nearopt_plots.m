clear
close all
clc

import ne_methods.*
import restart_schemes.fom_nesta
import restart_schemes.re_radial

% fix seed for debugging
rng(1)

%% TV minimization problem definition

% load image
X = double(imread('data/GPLU_phantom_512.png'))/255;

[N, ~] = size(X); % image size (assumed to be N by N)
nlevel = 1e-5;    % noise level

% near-optimal sampling mask
sample_rate = 0.125;
m = ceil(sample_rate*N*N);
var_hist = sa_inverse_square_law_hist_2d(N,1);
var_probs = sa_bernoulli_sampling_probs_2d(var_hist,m/2);
var_mask = binornd(ones(N,N),var_probs);
num_var_samples = sum(var_mask,'all');
uni_mask_cond = rand(N*N-num_var_samples,1) <= (m/2)/(N*N-num_var_samples);
uni_mask = zeros(N,N);
uni_mask(~var_mask) = uni_mask_cond;
mask = uni_mask | var_mask;
sample_idxs = find(mask);
m_exact = length(sample_idxs);

% flatten image
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

s = sum(abs(opW(x,0)) ~= 0); % sparsity level
f = @(z) norm(opW(z,0),1)/sqrt(s); % objective function
g = @(z) 0;      % gap function
kappa = 0;       % scalar factor for gap function

% here we project the zero vector onto the constraint set, resulting in z0
lmult = max(0,norm(y,2)/nlevel-1);
z0 = (lmult/((lmult+1)*c_A)).*opA(y,1);
eps0 = N*N;

% relative reconstruction error
eval_fns = {@(z) norm(z-x,2)};

nesta_cost = @(delta, eps) ceil(4*sqrt(2)*N*delta/eps);
nesta_algo = @(delta, eps, x_init, F) fom_nesta(...
    x_init, opA, c_A, y, opW, L_W, nesta_cost(delta,eps), nlevel, eps/(N*N), eval_fns, F);


%% Plotting parameters

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);


%% fixed alpha, beta

beta = 1;
alpha = logspace(2.3,2.75,10);
CMAP = linspecer(length(alpha));

t = 10000;
max_total_iters = 2000;

figure

for i=1:length(alpha)
    [~, VALS] = re_radial(...
    nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'beta',beta,'alpha',alpha(i),'total_iters',max_total_iters);
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

clear -regexp ^VALS;
clear legend_labels;

%% compare standard NESTA with radial-grid restart scheme

alpha1 = 475;
beta1 = 1;
beta2 = 1;

t = 100000;
max_total_iters = 30000;

figure

for i=1:3
    if i == 1
        [~, VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'alpha',alpha1,'beta',beta1,'total_iters',max_total_iters);
    elseif i == 2
        [~, VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'beta',beta2,'total_iters',max_total_iters);
    elseif i == 3
        [~, VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'total_iters',max_total_iters);
    end

    semilogy(VALS,'linewidth',2)
    
    hold on
end

mu = nlevel.*logspace(1,-2,4);

for i=1:length(mu)
    [~, NESTA_VALS] = fom_nesta(...
        z0, opA, c_A, y, opW, L_W, max_total_iters, nlevel, mu(i), eval_fns, @(x) 0);
    
    semilogy(NESTA_VALS,'linewidth',2);
    
    hold on
end

legend_labels = cell(3+length(mu),1);
legend_labels{1} = sprintf('$\\alpha = %s$, $\\beta = %s$', num2str(alpha1), num2str(beta1));
legend_labels{2} = sprintf('$\\alpha$-grid, $\\beta = %s$', num2str(beta2));
legend_labels{3} = '$(\alpha,\beta)$-grid';

for i=1:length(mu)
    legend_labels{3+i} = sprintf('no re., $\\mu = 10^{%s}$',num2str(log10(mu(i))));
end

legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,max(pd_ev_values)])

hold off

savefig(fullfile(dname,'nesta_comparison'))

clear -regexp ^VALS;
