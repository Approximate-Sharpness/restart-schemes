clear
close all
clc

% Comparison of Renegar & Grimmer's (best) synchronous restart scheme with
% ours, on a problem of sparse recovery from uniform subsampled Fourier
% measurements solved using NESTA.
%
% Plots of the minimum number of iterations needed to reach an epsilon
% precision in objective error versus epsilon are shown. One can see the
% log^2(1/epsilon) performance of R&G clearly.

import ne_methods.*
import restart_schemes.fom_nesta
import restart_schemes.re_radial
import restart_schemes.re_RG_best_sync

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
eval_fns = {@(z) norm(z-x,2), f};

nesta_cost = @(delta, eps) ceil(2*sqrt(N)*delta/eps);
nesta_algo = @(delta, eps, x_init, F) fom_nesta(...
    x_init, opA, c_A, y, opW, L_W, nesta_cost(delta,eps), nlevel, eps/N, eval_fns, F);

rg_iteration = @(state) nesta_iteration(state.z, state.q_v, state.n, opA, c_A, y, opW, L_W, nlevel, state.mu);
rg_initialize = @(z0, eps) nesta_initialize(z0, eps, N);


%% Plotting parameters

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);


%% compare our restart scheme with R&G's scheme

% our scheme
alpha1 = sqrt(m);
beta1 = 1;
beta2 = 1;
c1 = 2;

t = 25000;
max_total_iters = 10000;

for i=1:3
    if i == 1
        [~, ka_kb_VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'alpha',alpha1,'beta',beta1,'total_iters',max_total_iters);
    elseif i == 2
        [~, ua_kb_VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'a',exp(c1*beta2),'c1',c1,'alpha0',sqrt(m),'beta',beta2,'total_iters',max_total_iters);
    elseif i == 3
        [~, ua_ub_VALS] = re_radial(...
            nesta_algo,nesta_cost,f,g,kappa,z0,eps0,t,'a',exp(c1),'c1',c1,'alpha0',sqrt(m),'total_iters',max_total_iters);
    end
end

% R&G's scheme
rg_epsilon = logspace(-1,-8,50);
rg_state_values = cell(length(rg_epsilon),1);

for i=1:length(rg_epsilon)
    [state_values, xout] = re_RG_best_sync(rg_initialize, rg_iteration, f, z0, rg_epsilon(i), ceil(max_total_iters/ceil(-log2(rg_epsilon(i)))), eval_fns);
    rg_state_values{i} = state_values;
end

%% Plot objective error

opt_value = ka_kb_VALS(2,end);
iter_threshold_f = @(z, eps) max(z - opt_value,1e-15) > eps;

rg_iters = zeros(length(rg_epsilon),1);
ka_kb_iters = zeros(length(rg_epsilon),1);
ua_kb_iters = zeros(length(rg_epsilon),1);
ua_ub_iters = zeros(length(rg_epsilon),1);


for i=1:length(rg_epsilon)
    [~, rg_idx] = min(iter_threshold_f(rg_state_values{i}(2,:), rg_epsilon(i)));
    rg_iters(i) = rg_idx;

    [~, ka_kb_idx] = min(iter_threshold_f(ka_kb_VALS(2,:), rg_epsilon(i)));
    ka_kb_iters(i) = ka_kb_idx;

    [~, ua_kb_idx] = min(iter_threshold_f(ua_kb_VALS(2,:), rg_epsilon(i)));
    ua_kb_iters(i) = ua_kb_idx;

    [~, ua_ub_idx] = min(iter_threshold_f(ua_ub_VALS(2,:), rg_epsilon(i)));
    ua_ub_iters(i) = ua_ub_idx;
end

figure

semilogx(rg_epsilon, ka_kb_iters, '-', 'linewidth', 2)

hold on

semilogx(rg_epsilon, ua_kb_iters, '-', 'linewidth', 2)
semilogx(rg_epsilon, ua_ub_iters, '-', 'linewidth', 2)
semilogx(rg_epsilon, rg_iters,'--', 'linewidth', 2)

legend_labels = cell(4,1);

legend_labels{1} = sprintf('$\\alpha = %s$, $\\beta = %s$', num2str(alpha1), num2str(beta1));
legend_labels{2} = sprintf('$\\alpha$-grid, $\\beta = %s$', num2str(beta2));
legend_labels{3} = '$(\alpha,\beta)$-grid';
legend_labels{4} = sprintf('\\texttt{Sync||FOM}');

legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([1e-8, 1e-1])

hold off

savefig(fullfile(dname,'RG_iters_vs_epsilon'))


%% Function definitions

function state = nesta_initialize(z0, eps, N)
    state = struct('x', z0, 'z', z0, 'q_v', z0, 'n', 0, 'mu', eps/N);
end


function state = nesta_iteration(...
    z, q_v, n, opA, c_A, b, opW, L_W, nlvl, mu)
% Function computing one iteration of NESTA.
%

% compute x_n
grad = opW(z,0);
grad = restart_schemes.op_smooth_l1_gradient(grad,mu);
grad = mu/(L_W*L_W)*opW(grad,1);
q = z-grad;

dy = b-opA(q,0);
lam = max(0,norm(dy,2)/nlvl - 1);

x = lam/((lam+1)*c_A)*opA(dy,1) + q;

% compute v_n
alpha = (n+1)/2;
q_v = q_v-alpha*grad;
q = q_v;

dy = b-opA(q,0);
lam = max(0,norm(dy,2)/nlvl - 1);

v = lam/((lam+1)*c_A)*opA(dy,1) + q;

% compute z_{n+1}
tau = 2/(n+3);
z = tau*v+(1-tau)*x;

state = struct('x', x, 'z', z, 'q_v', q_v, 'n', n+1, 'mu', mu);

end