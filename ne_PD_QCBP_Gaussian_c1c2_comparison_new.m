clear
close all
clc

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
A = randn(m,N)/sqrt(m);
opA = @(z,ad) op_matrix_operator(A,z,ad);
L_A = norm(A,2); % Lipschitz constant

% measurement vector
e = randn(m,1);
b = A*x + nlevel*e/norm(e);

%% Restart scheme parameters

f = @(z) norm(z{1},1); % objective function
g = @(z) feasibility_gap(A*z{1}, b, nlevel); % gap function
kappa = sqrt(m); % scalar factor for gap function

x0 = zeros(N,1);
y0 = zeros(m,1);
x0y0 = {x0,y0};
opt_value = f({x}) + kappa.*g({x});
eps0 = f(x0y0) + kappa.*g(x0y0);

eval_fns = {@(z) norm(z{1}-x,2)};

pd_cost = @(delta, eps, xy_init) ceil(2*L_A*(kappa+norm(xy_init{2}))*delta/eps);
pd_algo = @(delta, eps, xy_init,F) fom_pd_QCBP(...
    xy_init{1}, xy_init{2}, delta/((kappa+norm(xy_init{2}))*L_A), (kappa+norm(xy_init{2}))/(delta*L_A), pd_cost(delta,eps,xy_init), opA, b, nlevel, eval_fns, F);


%% Plotting parameters

% x_axis_label = 'total iterations';
% y_axis_label = 'reconstruction error';

[~,fname,~] = fileparts(mfilename);
dname = sprintf('results/%s/', fname);
mkdir(dname);


%% Generate plots comparing performance of selecting c1 and c2

t = 50000;
max_total_iters = 2000;

% grid search alpha
%alpha0 = 2*sqrt(m);
alpha0 = sqrt(m);
beta = 1;
c1 = linspace(1,10,10);
CMAP1 = linspecer(length(c1));

figure
for i=1:length(c1)
    [~, GRID_VALS] = re_radial_pd(...
        pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'c1',c1(i),'alpha0',alpha0,'beta',beta,'total_iters',max_total_iters);
    semilogy(GRID_VALS,'linewidth',2,'color',CMAP1(i,:));
    hold on
end

legend_labels = cell(length(c1),1);
for i=1:length(c1)
    legend_labels{i} = strcat('$c_1 = $',sprintf(' %1.0f', c1(i)));
end
legend(legend_labels,'interpreter','latex','fontsize',14)
ax=gca; ax.FontSize=14;
xlim([0,max_total_iters]);  ylim([nlevel/4,10])
hold off
%savefig(fullfile(dname,sprintf('c1_comparison_bad_alpha0',c1(i))))
savefig(fullfile(dname,sprintf('c1_comparison_good_alpha0',c1(i))))


%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end
