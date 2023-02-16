% RE_RADIAL_PD Restart scheme with radial search modified for primal-dual. 
%
%   Implements a restart scheme modified to accelerate Chambolle-Pock
%   primal-dual iteration. This is an extension to RE_RADIAL that performs
%   restarts and tracks the dual variables for each instance. See the first
%   reference below.
%   
%   The sharpness constants are specified optionally, where for
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
%   alpha       - scaling sharpness parameter (defaults to grid search)
%   alpha0      - center for alpha grid search (defaults to 1)
%   a           - base for alpha points
%   beta        - exponent sharpness parameter (defaults to grid search)
%   beta0       - center for beta grid search (defaults to 1)
%   b           - base for beta points
%   r           - decay parameter (defaults to exp(-1))
%   c1          - alpha grid search schedule parameter (defaults to 2)
%   c2          - beta grid search schedule parameter (defaults to 2)
%   total_iters - stop after at some number of total iterations are 
%                 performed
%
% OUTPUT
% ======
%   result         - final iterate minimizing f + kappa*g
%   re_ev_values   - evaluation of eval_fns at each executed restart if
%                    defined for fom
%
% NOTES
% =====
%   The input 'fom' is expected to be a function handle that takes three
%   arguments: delta, eps, x_init. These are described as
%
%     - delta  : an upper bound of the l2-distance for the initial guess 
%                to a minimizer
%     - eps    : the target accuracy of 'fom'
%     - x_init : the initial vector for 'fom' (1x2 cell with initial
%                vectors for primal and dual problem)
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
%   The numerical experiments provided with this code give examples for how
%   to define C_fom and fom.
%   
%   Note that the radial ordering implementation here restricts the range
%   of which sharpness constants are considered, with exponent of maximum
%   absolute value proportional to floor(abs(log(eps))), where eps is 
%   machine epsilon.
%
% REFERENCES
% ==========
%   - "WARPd: A Linearly Convergent First-Order Primal-Dual Algorithm for 
%     Inverse Problems with Approximate Sharpness Conditions", Colbrook
%     (2022). doi:10.1137/21M1455000
%

function [result, ev_values] = re_radial_pd(...
    fom, C_fom, f, g, kappa, x0, eps0, restarts, varargin)

inp = inputParser;
validNumScalar = @(x) isnumeric(x) && isscalar(x);
validScaleScalar_ineq = @(x) validNumScalar(x) && x > 1;
validScaleScalar_eq = @(x) validNumScalar(x) && x >= 1;
validr = @(x) validNumScalar(x) && x < 1 && x > 0;
validPositiveScalar = @(x) validNumScalar(x) && x > 0;
addParameter(inp,'alpha',[],validPositiveScalar);
addParameter(inp,'alpha0',1,validPositiveScalar);
addParameter(inp,'a',exp(1),validScaleScalar_ineq);
addParameter(inp,'beta',[],validScaleScalar_eq);
addParameter(inp,'beta0',1,validScaleScalar_eq);
addParameter(inp,'b',exp(1),validScaleScalar_ineq);
addParameter(inp,'r',exp(-1),validr);
addParameter(inp,'c1',2,validScaleScalar_eq);
addParameter(inp,'c2',2,validScaleScalar_eq);
addParameter(inp,'total_iters',[],validPositiveScalar);
parse(inp,varargin{:});

total_iters = inp.Results.total_iters;
r = inp.Results.r;
a_exp = inp.Results.a;
b_exp = inp.Results.b;
c1 = inp.Results.c1;
c2 = inp.Results.c2;
alpha0 = inp.Results.alpha0;
beta0 = inp.Results.beta0;

grid_flags = [1,1];

if validNumScalar(inp.Results.alpha); grid_flags(1) = 0; end
if validNumScalar(inp.Results.beta); grid_flags(2) = 0; end

phi = restart_schemes.create_radial_order_schedule(restarts, a_exp, b_exp, c1, c2, grid_flags);

ij_tuples = unique(phi(:,1:2),'rows');

U = zeros(size(ij_tuples,1),1);
V = zeros(size(ij_tuples,1),1);

F = @(x) f(x) + kappa*g(x);
F_min_value = eps0;

ev_values = [];

x = x0;
m = 0;

DUAL={};
INDX=@(n) (n>0).*2*n +(n==0)+(n<0).*(2*abs(n)+1);

while true
    m = m + 1;
    
    if (m > restarts) || (~isempty(total_iters) && sum(V) > total_iters)
        break
    end
    
    i = phi(m,1); j = phi(m,2); k = phi(m,3);
    
    if k==1
        DUAL{INDX(i),j+1}=x0{2};
    end
    x{2}=DUAL{INDX(i),j+1};
        
    ij_ = find_idx_in_array(ij_tuples, [i,j]);
    
    if grid_flags(1); alpha = alpha0*a_exp^i; else; alpha = inp.Results.alpha; end
    if grid_flags(2); beta = beta0*b_exp^j; else; beta = inp.Results.beta; end
    
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
    
    cost = C_fom(delta, next_tol, x);
    if (V(ij_)+cost) <= k
        [z, values] = fom(delta, next_tol, x, F);
        ev_values = [ev_values, values];
        
        F_next_value = F(z); % perform argmin
        if F_next_value < F_min_value
            F_min_value = F_next_value;
            x = z;
            DUAL{INDX(i),j+1}=z{2};
        end
        
        V(ij_) = V(ij_) + cost;
        U(ij_) = U(ij_) + 1;
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
