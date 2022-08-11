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
%   eps0   - an upper bound for f(x0) + kappa*g(x0)
%   t      - number of restart iterations
%
% OPTIONAL PARAMETERS
% ===================
%   alpha  - scaling sharpness constant
%   beta   - exponent sharpness constant
%   eta    - additive sharpness constant
%
% OUTPUT
% ======
%   result         - final iterate minimizing f + kappa*g
%   inner_o_values - cell of o_values output from each fom instance
%   inner_c_values - cell of c_values output from each fom instance
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

function [result, inner_o_values, inner_c_values] = re_radial_search(...
    fom, C_fom, f, g, kappa, x0, eps0, t, varargin)

inp = inputParser;
validNumScalar = @(x) isnumeric(x) && isscalar(x);
addParameter(inp,'alpha',[],validNumScalar);
addParameter(inp,'beta',[],validNumScalar);
addParameter(inp,'eta',[],validNumScalar);
parse(inp,varargin{:});

grid_flags = [1,1,1];

if validNumScalar(inp.Results.alpha); grid_flags(1) = 0; end
if validNumScalar(inp.Results.beta); grid_flags(2) = 0; end
if validNumScalar(inp.Results.eta); grid_flags(3) = 0; end

phi = restart_schemes.create_radial_order_schedule(t, grid_flags);

ijk_tuples = unique(phi(:,1:3),'rows');

r = exp(-1);
U = zeros(size(ijk_tuples,1),1);
V = zeros(size(ijk_tuples,1),1);

x = x0;

F = @(x) f(x) + kappa*g(x);
F_min_value = F(x0);

inner_o_values = cell(t,1);
inner_c_values = cell(t,1);

for m=1:t
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
        fprintf('Executing restart with (i,j,k,l) == (%d,%d,%d,%d)\n',i,j,k,l)
        [z, o_values, c_values] = fom(delta, next_eps, x);
        inner_o_values{m} = o_values;
        inner_c_values{m} = c_values;
        F_next_value = F(z);
        if F_next_value < F_min_value
            F_min_value = F_next_value;
            x = z;
        end
        
        V(ijk_) = V(ijk_) + C_fom(delta, next_eps);
        U(ijk_) = U(ijk_) + 1;
    end  
end

fprintf('Total iterations executed: %d\n', sum(V))
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