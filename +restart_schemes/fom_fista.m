% FOM_FISTA  Fast Iterative-Shrinkage Thresholding Algorithm (FISTA)
%
%   An accelerated first-order method to solve the LASSO problem.
%
% INPUT
% =====
%   x0        - initial guess
%   opA       - LASSO fidelity linear operator
%   b         - LASSO fidelity vector
%   lam       - LASSO penalty term
%   step      - step-size per iteration
%   num_iters - number of iterations
%   eval_obj  - flag to store objective function and constraint 
%               evaluations
%
% OUTPUT
% ======
%   result    - approximate minimizer of LASSO
%   o_values  - objective function evaluations of primal iterates
%   c_values  - constraint evaluations of primal iterates
%
% NOTES
% =====
%   The input opA should be a function handle, with the first argument a
%   compatible input and the second argument a boolean flag to enable using
%   the transpose of the linear operator (when the flag is nonzero). The 
%   purpose of expressing opA in this way is to allow for efficient
%   implementations (e.g. fast wavelet transform).
%
% REFERENCES
% ==========
%   - "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse 
%     Problems", Beck & Teboulle (2009). [doi:10.1137/080716542]
%

function [result, o_values, c_values] = fom_fista(x0, opA, b, lam, step, num_iters, eval_obj)

x = x0;
t = 1;
z = x0;

o_values = cell(num_iters,1);
c_values = cell(num_iters,1);


for i=1:num_iters
    
    q = z - step.*(opA(opA(z,0)-b,1));
    x_next = max(abs(q)-(lam*step),0).*sign(q);
    t_next = 0.5*(1+sqrt(1+4*t.^2));

    z = x_next + ((t-1)/t_next)*(x_next-x);
    
    x = x_next;
    t = t_next;
    
    if eval_obj
        o_values{i} = norm(x,1);
        c_values{i} = norm(opA(x,0)-b,2);
    end
    
result = x;

end