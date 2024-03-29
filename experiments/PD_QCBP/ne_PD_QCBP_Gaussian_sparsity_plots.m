clear
close all
clc

% Comparison of various restart schemes with the sparsity of the ground 
% truth vector varied in the sparse recovery problem.

import ne_methods.op_matrix_operator 
import restart_schemes.fom_pd_QCBP
import restart_schemes.re_radial_pd

% fix seed for debugging
rng(1);

%% QCBP problem definition

N = 128;          % s-sparse vector size
s = linspace(12,22,11);           % sparsity
m = 60;           % measurements
nlevel = 1e-6;    % noise level

x = zeros(N,length(s));
for i=1:length(s)
    x(1:s(i),i) = randn(s(i),1);
    x(:,i) = x(randperm(N),i);
end

e = randn(m,1);
e = nlevel*e/norm(e);

% measurement matrix (Gaussian random)
[A, L_A] = generate_gaussian_matrix(m,N);
opA = @(z,ad) op_matrix_operator(A,z,ad);

%% Restart scheme parameters

f = @(z) norm(z{1},1); % objective function
kappa = sqrt(m); % scalar factor for gap function
alpha0 = sqrt(m); % shift in alpha

x0 = zeros(N,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};

%% Plotting parameters

% x_axis_label = 'total iterations';
% y_axis_label = 'reconstruction error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);

%% fixed alpha and fixed beta
beta = 1;
alpha = sqrt(m);
CMAP = linspecer(length(s));

t = 4000;
total_iters = 1000;

figure
for i=1:length(s)
    
    eval_fns = {@(z) norm(z{1}-x(:,i),2)};
    
    b = A*x(:,i) + e;
    g = @(z) feasibility_gap(A*z{1}, b, nlevel);
    eps0 = f(x0y0) + kappa.*g(x0y0);
    
    pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
    pd_algo = @(delta, eps, xy_init,F) fom_pd_QCBP(...
        xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);

    scheme = @(t,varargin) re_radial_pd(pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha0',alpha0,varargin{:});

    [~, VALS] = scheme(t,'alpha',alpha,'beta',beta,'total_iters',total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(s),1);
for i=1:length(s)
    legend_labels{i} = sprintf('$s = %s$', num2str(s(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([min(nlevel)/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;


%% fixed alpha and search over beta
alpha = sqrt(m);

t = 20000;
total_iters = 5000;

figure
for i=1:length(s)
    
    eval_fns = {@(z) norm(z{1}-x(:,i),2)};
    
    b = A*x(:,i) + e;
    g = @(z) feasibility_gap(A*z{1}, b, nlevel);
    eps0 = f(x0y0) + kappa.*g(x0y0);
    
    pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
    pd_algo = @(delta, eps, xy_init,F) fom_pd_QCBP(...
        xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);

    scheme = @(t,varargin) re_radial_pd(pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha0',alpha0,varargin{:});

    [~, VALS] = scheme(t,'alpha',alpha,'total_iters',total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(s),1);
for i=1:length(s)
    legend_labels{i} = sprintf('$s = %s$', num2str(s(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([min(nlevel)/4,max(VALS)])
hold off
savefig(fullfile(dname,'fixed_alpha_search_beta'))

clear -regexp ^VALS;
clear legend_labels;


%% fixed beta and search over alpha
beta = 1;
c1 = 2;

t = 10000;
total_iters = 3000;

figure
for i=1:length(s)
    
    eval_fns = {@(z) norm(z{1}-x(:,i),2)};
    
    b = A*x(:,i) + e;
    g = @(z) feasibility_gap(A*z{1}, b, nlevel);
    eps0 = f(x0y0) + kappa.*g(x0y0);
    
    pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
    pd_algo = @(delta, eps, xy_init,F) fom_pd_QCBP(...
        xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);

    scheme = @(t,varargin) re_radial_pd(pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha0',alpha0,varargin{:});

    [~, VALS] = scheme(t,'c1',c1,'a',exp(c1*beta),'beta',beta,'total_iters',total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(s),1);
for i=1:length(s)
    legend_labels{i} = sprintf('$s = %s$', num2str(s(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([min(nlevel)/4,max(VALS)])
hold off
savefig(fullfile(dname,'search_alpha_fixed_beta'))

clear -regexp ^VALS;
clear legend_labels;

%% grid search on both alpha and beta

t = 10000;
total_iters = 3000;
c1 = 2;

figure
for i=1:length(s)
    
    eval_fns = {@(z) norm(z{1}-x(:,i),2)};
    
    b = A*x(:,i) + e;
    g = @(z) feasibility_gap(A*z{1}, b, nlevel);
    eps0 = f(x0y0) + kappa.*g(x0y0);
    
    pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
    pd_algo = @(delta, eps, xy_init,F) fom_pd_QCBP(...
        xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);

    scheme = @(t,varargin) re_radial_pd(pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'alpha0',alpha0,varargin{:});

    [~, VALS] = scheme(t,'c1',c1,'a',exp(c1),'total_iters',total_iters);
    semilogy(VALS,'linewidth',2,'color',CMAP(i,:));
    hold on
end

legend_labels = cell(length(s),1);
for i=1:length(s)
    legend_labels{i} = sprintf('$s = %s$', num2str(s(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,total_iters]);  ylim([min(nlevel)/4,max(VALS)])
hold off
savefig(fullfile(dname,'search_alpha_search_beta'))

clear -regexp ^VALS;
clear legend_labels;


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