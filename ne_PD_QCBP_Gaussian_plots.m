clear
close all
clc

% Performance of various restart schemes with different approximate 
% sharpness parameters on a sparse recovery problem.

import ne_methods.op_matrix_operator 
import restart_schemes.fom_pd_QCBP
import restart_schemes.re_radial_pd

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

% measurement matrix (Gaussian random)
[A, L_A] = generate_gaussian_matrix(m,N);
opA = @(z,ad) op_matrix_operator(A,z,ad);
[b, e] = generate_noisy_measurements(A*x,nlevel);

%% Restart scheme parameters

f = @(z) norm(z{1},1); % objective function
g = @(z) feasibility_gap(A*z{1}, b, nlevel); % gap function
kappa = sqrt(m); % scalar factor for gap function
alpha0 = sqrt(m); % shift in alpha

x0 = zeros(N,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};
eps0 = f(x0y0) + kappa.*g(x0y0);

eval_fns = {@(z) norm(z{1}-x,2)};

pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
pd_algo = @(delta, eps, xy_init,F) fom_pd_QCBP(...
    xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);

scheme = @(t,varargin) re_radial_pd(pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha0',alpha0,varargin{:});

%% Plotting parameters

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha and fixed beta
beta = 1;
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 4000;
total_iters = 2000;

figure
for i=1:length(alpha)
    [~, VALS] = scheme(t,'alpha',alpha(i),'beta',beta,'total_iters',total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([nlevel/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% fixed alpha and search over beta
alpha = logspace(0.2,2,10);
CMAP = linspecer(length(alpha));

t = 100000;
total_iters = 5000;

figure
for i=1:length(alpha)
    [~, VALS] = scheme(t,'alpha',alpha(i),'total_iters',total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(alpha),1);
for i=1:length(alpha)
    legend_labels{i} = strcat('$\log_{10}(\alpha) = $',sprintf(' %s', num2str(log10(alpha(i)))));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([nlevel/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_search_beta'))

clear -regexp ^VALS;
clear legend_labels;


%% fixed beta and search over alpha
beta = 1:0.5:3;
CMAP = linspecer(length(beta));

t = 10000;
total_iters = 3000;
c1 = 2;

figure
for i=1:length(beta)
    [~, VALS] = scheme(t,'c1',c1,'a',exp(c1*beta(i)),'beta',beta(i),'total_iters',total_iters);
    semilogy(VALS,'linewidth',2','color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(beta),1);
for i=1:length(beta)
    legend_labels{i} = strcat('$\beta = $',sprintf(' %1.1f', beta(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([nlevel/4,max(VALS)])
hold off
savefig(fullfile(dname,'search_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% compare standard PD with radial-grid restart scheme

c1 = 2;
beta2 = 1;
alpha1 = sqrt(m);
beta1 = 1;

t = 20000;
total_iters = 5000;

figure

[~, pd_ev_values] = fom_pd_QCBP(...
    x0, y0, eps0/L_A, 1/(eps0*L_A), total_iters, opA, b, nlevel, eval_fns,@(x) f(x)+kappa*g(x));

for i=1:3
    if i == 1
        [~, VALS] = scheme(t,'alpha',alpha1,'beta',beta1,'total_iters',total_iters);
    elseif i == 2
        [~, VALS] = scheme(t,'c1',c1,'a',exp(c1*beta2),'beta',beta2,'total_iters',total_iters);
    elseif i == 3
        [~, VALS] = scheme(t,'c1',c1,'a',exp(c1),'total_iters',total_iters);
    end
    semilogy(VALS,'linewidth',2);
    hold on
end

semilogy(pd_ev_values,'linewidth',2,'linestyle','--');

legend_labels = cell(3,1);
legend_labels{1} = strcat('$\alpha = $',sprintf(' %1.1f,', alpha1),' $\beta = $',sprintf(' %1.1f', beta1));
legend_labels{2} = strcat('$\alpha$-grid,',' $\beta = $',sprintf(' %1.1f', beta2));
legend_labels{3} = '$(\alpha,\beta)$-grid';
legend_labels{4} = 'no restarts';
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([nlevel/4,max(pd_ev_values)])

hold off

savefig(fullfile(dname,'pd_comparison'))

clear -regexp ^VALS;



%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
    dist = norm(z-center,2);
    out = max(dist-rad,0);
end

function [A, L_A] = generate_gaussian_matrix(m, N)
    A = randn(m,N)/sqrt(m);
    L_A = norm(A,2);
end

function [b, e] = generate_noisy_measurements(b_exact, nlvl)
    m = length(b_exact);
    e = randn(m,1);
    b = b_exact + nlvl*e/norm(e);
end
