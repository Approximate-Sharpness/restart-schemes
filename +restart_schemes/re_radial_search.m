% RE_RADIAL_SEARCH Restart scheme with radial search for sharpness. 
%
%   Implements a general restart scheme for a first-order optimization 
%   algorithm. The sharpness constants are specified optionally, where for
%   those not specified, a logarithmic grid search for the unspecified
%   constants is augmented to the restart scheme. The search uses a radial 
%   ordering of instance execution. [TO DO ... / REF]
%
% REQUIRED INPUT
% ==============
%   fom    - function handle of a first-order method (see NOTES)
%   C_fom  - cost function of fom (see NOTES)
%   f      - scalar objective function
%   g      - scalar feasibility gap function
%   kappa  - feasibility gap parameter
%   x0     - initial guess 
%   restarts    - number of restarts to perform
%
% OPTIONAL PARAMETERS
% ===================
%   alpha       - scaling sharpness constant
%   beta        - exponent sharpness constant
%   eta         - additive sharpness constant
%   eval_fns    - cell array of function handles to evaluate on each 
%                 iterate
%   total_iters - stop after at some number of total iterations are 
%                 performed
%
% OUTPUT
% ======
%   result         - final iterate minimizing f + kappa*g
%   re_ev_values   - evaluation of eval_fns at each executed restart
%   re_inner_iters - number of inner iterations at each executed restart
%
% NOTES
% =====
%   The input 'fom' is expected to be a function handle that takes three
%   arguments: delta, eps, x_init. These are described as
%
%     - delta  : an upper bound of the l2-distance for the initial guess 
%                to a minimizer
%     - eps    : the target accuracy of 'fom'
%     - x_init : the initial vector for 'fom'
%
%   The output of 'fom' should be: result, o_values, c_values. The result
%   is the final output iterate achieving objective error eps, and cells
%   o_values and c_values containing evaluations of the objective function 
%   and constraints, respectively.
%
%   The input 'C_fom' is a function handle taking in delta and eps, just
%   like in 'fom', and outputs an integer specifying the number of
%   iterations needed for 'fom' to achieve precision eps with initial
%   distance delta.
%   
%   See [TO DO ... / REF] for examples of how to define 'fom' and 'C_fom'.
%
%   
%   Note that the radial ordering implementation here restricts the range
%   of which sharpness constants are considered, with exponent of maximum
%   absolute value equal to 32. Thus, for example, the smallest eta value
%   tested is exp(-32).
%
% REFERENCES
% ==========
%   - TO DO ...
%

function [result, re_ev_values, re_inner_iters] = re_radial_search(...
    fom, C_fom, f, g, kappa, x0, restarts, varargin)

inp = inputParser;
validNumScalar = @(x) isnumeric(x) && isscalar(x);
validPositiveScalar = @(x) validNumScalar(x) && x > 0;
validFnHandles = @(x) iscell(x) && all(cellfun(@(f) isa(f,'function_handle'),x));
addParameter(inp,'alpha',[],validNumScalar);
addParameter(inp,'beta',[],validNumScalar);
addParameter(inp,'eta',[],validNumScalar);
addParameter(inp,'eval_fns',[],validFnHandles);
addParameter(inp,'total_iters',[],validPositiveScalar);
parse(inp,varargin{:});

total_iters = inp.Results.total_iters;
eval_fns = inp.Results.eval_fns;

grid_flags = [1,1,1];

if validNumScalar(inp.Results.alpha); grid_flags(1) = 0; end
if validNumScalar(inp.Results.beta); grid_flags(2) = 0; end
if validNumScalar(inp.Results.eta); grid_flags(3) = 0; end

phi = restart_schemes.create_radial_order_schedule(restarts, grid_flags);

ijk_tuples = unique(phi(:,1:3),'rows');

r = exp(-1);
U = zeros(size(ijk_tuples,1),1);
V = zeros(size(ijk_tuples,1),1);

F = @(x) f(x) + kappa*g(x);
eps0 = F(x0);
F_min_value = eps0;

re_ev_values = cell(restarts+1,1);
re_inner_iters = cell(restarts+1,1);

re_ev_values{1} = zeros(length(eval_fns),1);
for fidx=1:length(eval_fns)
    re_ev_values{1}(fidx) = eval_fns{fidx}(x0);
end
re_inner_iters{1} = 0;

x = x0;
m = 0;

while true
    m = m + 1;
    
    if (m > restarts) || (~isempty(total_iters) && sum(V) > total_iters)
        re_ev_values = re_ev_values(1:m-1);
        re_inner_iters = re_inner_iters(1:m-1);
        break
    end
    
    i = phi(m,1); j = phi(m,2); k = phi(m,3); l = phi(m,4);
    
    ijk_ = find_idx_in_array(ijk_tuples, [i,j,k]);
    
    if grid_flags(1); alpha = exp(i); else; alpha = inp.Results.alpha; end
    if grid_flags(2); beta = exp(j); else; beta = inp.Results.beta; end
    if grid_flags(3); eta = exp(k); else; eta = inp.Results.eta; end
    
    if U(ijk_) == 0
        eps = eps0;
    else
        eps = r^(U(ijk_))*eps0 + r*(1-r^(U(ijk_)))/(1-r)*eta;
    end
    
    next_eps = r*(eps + eta);
    
    if grid_flags(2) % if beta grid search is enabled
        if (eps + eta) > alpha
            delta = ((eps + eta)/alpha)^(min(exp(1)/beta,1));
        else
            delta = ((eps + eta)/alpha)^(1/beta);
        end
    else % otherwise, beta grid search is disabled
        delta = ((eps + eta)/alpha)^(1/beta);
    end
    
    if (V(ijk_)+C_fom(delta, next_eps)) <= l
        [z, ~] = fom(delta, next_eps, x);
        
        if all(~grid_flags)
            x = z; % do not use argmin if using all fixed sharpness constants
        else
            F_next_value = F(z);
            if F_next_value < F_min_value
                F_min_value = F_next_value;
                x = z;
            end
        end
        
        re_ev_values{m+1} = zeros(length(eval_fns),1);
        for fidx=1:length(eval_fns)
            re_ev_values{m+1}(fidx) = eval_fns{fidx}(x);
        end
        re_inner_iters{m+1} = C_fom(delta, next_eps);
        
        V(ijk_) = V(ijk_) + C_fom(delta, next_eps);
        U(ijk_) = U(ijk_) + 1;
    end  
end

result = x;

end



function idx = find_idx_in_array(A, target_row)
% finds the right row for U and V given a tuple (i,j,k)
n = size(A,1);
for i=1:n
    if all(A(i,:) == target_row)
        idx = i;
        break
    end
end

end