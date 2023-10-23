clear
close all
clc

% Performance of various restart schemes with different approximate 
% sharpness parameters on a sparse recovery problem.

import ne_methods.*
import restart_schemes.fom_nesta
import restart_schemes.re_radial

% fix seed for debugging
rng(1)

%% QCBP problem definition

N = 128;          % s-sparse vector size
s = 15;           % sparsity
m = 60;           % measurements
nlevel = 1e-6;    % noise level

% s-sparse vector
x = zeros(N,1);
x(1:s) = randn(s,1);
x = x(randperm(N));

% uniform random sampling mask
sample_rate = m / N;
mask = rand(N,1) <= sample_rate;
sample_idxs = find(mask);
m_exact = length(sample_idxs);

% measurement matrix (subsampled Fourier)
opA = @(z,ad) sqrt(N/m)*op_fourier_1d(z, ad, N, sample_idxs);
c_A = N/m;

% measurement vector
e = randn(m_exact,1) + 1i*randn(m_exact,1);
y = opA(x,0) + nlevel*e/norm(e);

% analysis matrix (identity for QCBP)
W = eye(N);
opW = @(z,ad) op_matrix_operator(W,z,ad);
L_W = 1;

%% Restart scheme parameters

f = @(z) norm(opW(z,0),1); % objective function
g = @(z) 0;      % gap function
kappa = 0;       % scalar factor for gap function

% here we project the zero vector onto the constraint set, resulting in z0
lmult = max(0,norm(y,2)/nlevel-1);
z0 = (lmult/((lmult+1)*c_A)).*opA(y,1);
eps0 = f(z0);

% relative reconstruction error
eval_fns = {@(z) norm(z-x,2)};

nesta_cost = @(delta, eps) ceil(2*sqrt(N)*delta/eps);
nesta_algo = @(delta, eps, x_init, F) fom_nesta(...
    x_init, opA, c_A, y, opW, L_W, nesta_cost(delta,eps), nlevel, eps/N, eval_fns, F);

%% Plotting parameters

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha, beta

beta = 1;
alpha = logspace(0.5,1.5,11);
CMAP = linspecer(length(alpha));

t = 5000;
max_total_iters = 500;

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

alpha1 = 10^(0.9);
beta1 = 1;
beta2 = 1;
c1 = 2;

t = 20000;
max_total_iters = 3000;

figure

for i=1:3
    if i == 1
        [~, VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'alpha',alpha1,'beta',beta1,'total_iters',max_total_iters);
    elseif i == 2
        [~, VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'a',exp(c1*beta2),'c1',c1,'alpha0',sqrt(m),'beta',beta2,'total_iters',max_total_iters);
    elseif i == 3
        [~, VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'a',exp(c1),'c1',c1,'alpha0',sqrt(m),'total_iters',max_total_iters);
    end

    semilogy(VALS,'linewidth',2)
    
    hold on
end

legend_labels = cell(3,1);
legend_labels{1} = sprintf('$\\alpha = %s$, $\\beta = %s$', num2str(alpha1), num2str(beta1));
legend_labels{2} = sprintf('$\\alpha$-grid, $\\beta = %s$', num2str(beta2));
legend_labels{3} = '$(\alpha,\beta)$-grid';

legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,max(VALS)])

hold off

savefig(fullfile(dname,'nesta_comparison'))

clear -regexp ^VALS;
