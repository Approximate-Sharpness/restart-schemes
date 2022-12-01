clear
close all
clc

% Comparison of selecting grid search parameters c1 and c2. A plot is
% generated for each choice of c1 with varying c2.

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
alpha0 = 2*sqrt(m);

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

c1 = [2,4,6,8,10,12];
c2 = [2,4,6,8,10,12];

CMAP = linspecer(length(c2));

t = 10000;
max_total_iters = 1500;

for i=1:length(c1)
    figure
    for j=1:length(c2)
        [~, GRID_VALS] = re_radial_pd(...
            pd_algo,pd_cost,f,g,kappa,x0y0,eps0,t,'a',exp(c1(i)),'alpha0',alpha0,'c1',c1(i),'c2',c2(j),'total_iters',max_total_iters);
        semilogy(GRID_VALS,'linewidth',2,'color',CMAP(j,:));
        hold on
        end
    legend_labels = cell(length(c2),1);
    for j=1:length(c2)
        legend_labels{j} = strcat('$c_2 = $',sprintf(' %1.0f', c2(j)));
    end
    legend(legend_labels,'interpreter','latex','fontsize',14)
    ax=gca; ax.FontSize=14;
    xlim([0,max_total_iters]);  ylim([nlevel/4,10])
    hold off
    savefig(fullfile(dname,sprintf('c1c2_comparison_c1_%d',c1(i))))
end


%% Additional functions specific to the experiment

% Feasibility gap function handle
function out = feasibility_gap(z, center, rad)
dist = norm(z-center,2);
out = max(dist-rad,0);
end
