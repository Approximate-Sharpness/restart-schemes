% RE_RADIAL_SEARCH Restart scheme with radial search for sharpness. 
%
%   Implements a general restart scheme for a first-order optimization 
%   algorithm. The sharpness constants are specified optionally, where for
%   those not specified, a logarithmic grid search for the unspecified
%   constants is augmented to the restart scheme. The search uses a radial 
%   ordering of instance execution. 
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
%   absolute value equal to floor(abs(log(eps))), where eps is machine
%   epsilon.
%
% REFERENCES
% ==========
%   - TO DO ...
%

function [result, re_ev_values, re_inner_iters, VALS] = re_radial_search(...
    fom, C_fom, f, g, kappa, x0, eps0, restarts, varargin)

inp = inputParser;
validNumScalar = @(x) isnumeric(x) && isscalar(x);
validPositiveScalar = @(x) validNumScalar(x) && x > 0;
validScaleScalar = @(x) validNumScalar(x) && x > 1;
validr = @(x) validNumScalar(x) && x < 1 && x > 0;
validFnHandles = @(x) iscell(x) && all(cellfun(@(f) isa(f,'function_handle'),x));
addParameter(inp,'alpha',[],validNumScalar);
addParameter(inp,'a',exp(1),validScaleScalar);
addParameter(inp,'beta',[],validNumScalar);
addParameter(inp,'b',exp(1),validScaleScalar);
addParameter(inp,'r',exp(-1),validr);
addParameter(inp,'eval_fns',[],validFnHandles);
addParameter(inp,'total_iters',[],validPositiveScalar);
parse(inp,varargin{:});

total_iters = inp.Results.total_iters;
eval_fns = inp.Results.eval_fns;

grid_flags = [1,1];

if validNumScalar(inp.Results.alpha); grid_flags(1) = 0; end
if validNumScalar(inp.Results.beta); grid_flags(2) = 0; end

phi = restart_schemes.create_radial_order_schedule(restarts, grid_flags);

ij_tuples = unique(phi(:,1:2),'rows');

r = inp.Results.r;
a_exp = inp.Results.a;
b_exp = inp.Results.b;
U = zeros(size(ij_tuples,1),1);
V = zeros(size(ij_tuples,1),1);

F = @(x) f(x) + kappa*g(x);
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

DUAL={};
INDX=@(n) (n>0).*2*n +(n==0)+(n<0).*(2*abs(n)+1);
VALS=zeros(size(eval_fns,1),1);

while true
    m = m + 1;
    
    if (m > restarts) || (~isempty(total_iters) && sum(V) > total_iters)
        re_ev_values = re_ev_values(1:m-1);
        re_inner_iters = re_inner_iters(1:m-1);
        break
    end
    
    i = phi(m,1); j = phi(m,2); k = phi(m,3);
    if k==1
        DUAL{INDX(i),j+1}=x0{2};
    end
    x{2}=DUAL{INDX(i),j+1};
        
    
    ij_ = find_idx_in_array(ij_tuples, [i,j]);
    
    
    if grid_flags(2); beta = b_exp^j; else; beta = inp.Results.beta; end
    if grid_flags(1); alpha = a_exp^i*10; else; alpha = inp.Results.alpha; end
    
    tol = r^(U(ij_))*eps0;
    
    next_tol = max(r*tol,10*eps); % do not go below machine epsilon
    
    if grid_flags(2) % if beta grid search is enabled
        if (2*tol) > alpha
            delta = (2*tol/alpha)^(min(b_exp/beta,1));
        else
            delta = (2*tol/alpha)^(1/beta);
        end
    else % otherwise, beta grid search is disabled
        delta = (2*tol/alpha)^(1/beta);
    end
    
    delta = max(delta,10*eps); % do not go below machine epsilon
    
    if (V(ij_)+C_fom(delta, next_tol, x)) <= k
        CCC=C_fom(delta, next_tol, x);
        [z,ev_values2] = fom(delta, next_tol, x, F);
        if ~isempty(ev_values2)
            VALS=[VALS,ev_values2];
        end
        F_next_value = F(z); % perform argmin
        if F_next_value < F_min_value
            F_min_value = F_next_value;
            x = z;
            DUAL{INDX(i),j+1}=z{2};
        else
            for jjj=1:size(VALS,1)
                VALS(jjj,end-size(ev_values2,2):end)=VALS(jjj,end-size(ev_values2,2));
            end
        end
        
        re_ev_values{m+1} = zeros(length(eval_fns),1);
        for fidx=1:length(eval_fns)
            re_ev_values{m+1}(fidx) = eval_fns{fidx}(x);
        end
        re_inner_iters{m+1} = CCC;
        
        V(ij_) = V(ij_) + CCC;
        U(ij_) = U(ij_) + 1;
    end  
end

result = x;
VALS=VALS(:,2:end);
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
