% RE_FIXED_CONSTS_NEW Restart scheme with fixed sharpness constants. 
%
%   IMPORTANT! THIS IMPLEMENTS A NEW VERSION OF RESTART SCHEMES THAT
%   REMOVES ETA AS A SHARPNESS PARAMETER IN THE SCHEME!
%
%   Description [TO DO ... / REF]
%
% REQUIRED INPUT
% ==============
%   fom    - function handle of a first-order method (see NOTES)
%   C_fom  - cost function of fom (see NOTES)
%   f      - scalar objective function
%   g      - scalar feasibility gap function
%   kappa  - feasibility gap parameter
%   x0     - initial guess
%   eps0   - an upper bound for f(x0) + kappa*g(x0)
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

function [result, re_ev_values, re_inner_iters] = re_fixed_consts_new(...
    fom, C_fom, f, g, kappa, x0, eps0, restarts, alpha, beta, varargin)

inp = inputParser;
validPositiveScalar = @(x) isnumeric(x) && isscalar(x) && x > 0;
validFnHandles = @(x) iscell(x) && all(cellfun(@(f) isa(f,'function_handle'),x));
addParameter(inp,'eval_fns',[],validFnHandles);
addParameter(inp,'total_iters',[],validPositiveScalar);
parse(inp,varargin{:});

total_iters = inp.Results.total_iters;
eval_fns = inp.Results.eval_fns;

mach_eps = eps;

F = @(x) f(x) + kappa*g(x);

F_min_value = eps0;

r = exp(-1);
x = x0;
beps = eps0;

re_ev_values = cell(restarts+1,1);
re_inner_iters = cell(restarts+1,1);

re_ev_values{1} = zeros(length(eval_fns),1);
for fidx=1:length(eval_fns)
    re_ev_values{1}(fidx) = eval_fns{fidx}(x0);
end
re_inner_iters{1} = 0;

m = 0;
T = 0;

while true
    m = m + 1;
    
    if (m > restarts) || (~isempty(total_iters) && T > total_iters)
        re_ev_values = re_ev_values(1:m-1);
        re_inner_iters = re_inner_iters(1:m-1);
        break
    end
    
    next_beps = max(r*beps,mach_eps);
    delta = (2*beps/alpha)^(1/beta);
    fprintf('total iters: %d\n', C_fom(delta, next_beps))
    z = fom(delta, next_beps, x);

    F_next_value = F(z);
    if F_next_value < F_min_value
        F_min_value = F_next_value;
        x = z;
    end
    %x = z;
    
    re_ev_values{m+1} = zeros(length(eval_fns),1);
    for fidx=1:length(eval_fns)
        re_ev_values{m+1}(fidx) = eval_fns{fidx}(x);
    end
    re_inner_iters{m+1} = C_fom(delta, next_beps);
    T = T + C_fom(delta, next_beps);
    
    beps = next_beps;
end

result = x;

end
