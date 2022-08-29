% FOM_NESTA NESTerov's Algorithm to solve analysis-QCBP.
% 
%   An accelerated projected gradient descent algorithm to solve
%   analysis quadratically constrained basis pursuit (aQCBP).
%
% INPUT
% =====
%   z0        - initial vector for NESTA 
%   opA       - QCBP fidelity linear operator (see NOTES)
%   c_A       - constant c such that A*A' = c.*I (see NOTES)
%   b         - QCBP fidelity vector
%   opW       - QCBP analysis linear operator
%   L_W       - operator 2-norm of opW
%   num_iters - number of iterations
%   nlvl      - noise level
%   mu        - smoothing parameter
%   eval_fns  - cell array of function handles to evaluate on each iterate
%               (assign the empty array [] to disable)
%
% OUTPUT
% ======
%   result    - approximate minimizer of aQCBP
%   ev_values - eval_fns function evaluations of iterates
%
% NOTES
% =====
%   To use NESTA correctly, the linear operator A must satisfy A*A' = c.*I 
%   where I is the identity map and c > 0 is a constant. In other words,
%   representing A as a matrix, its rows are mutually orthonormal up to
%   a constant factor.
%
%   opA and opW should be function handles representing linear operators,
%   each with their first argument a compatible input and their second 
%   argument a boolean flag to enable using the adjoint of the linear 
%   operator (when the flag is nonzero).
%
%   The purpose of expressing opA and opW in this way is to allow for 
%   efficient implementations (e.g. fast wavelet transform).
%
%   Additionally, this algorithm is implemented to handle complex-valued
%   data, so opA can be say, the discrete Fourier transform.
%
% REFERENCES
% ==========
%   - TO DO ...
%

function [result, ev_values] = fom_nesta(z0, opA, c_A, b, opW, L_W, num_iters, nlvl, mu, eval_fns)

z = z0;
q_v = z0;

ev_values = zeros(length(eval_fns),num_iters);

for n=0:num_iters-1
    % compute x_n
    grad = opW(z,0);
    grad = restart_schemes.op_smooth_l1_gradient(grad,mu);
    grad = mu/(L_W*L_W)*opW(grad,1);
    q = z-grad;
    
    dy = b-opA(q,0);
    lam = max(0,norm(dy,2)/nlvl - 1);
    
    x = lam/((lam+1)*c_A)*opA(dy,1) + q;
    
    
    if ~isempty(eval_fns)
        for fidx=1:length(eval_fns)
            ev_values(fidx,n+1) = eval_fns{fidx}(x);
        end
    end
    
    % compute v_n
    alpha = (n+1)/2;
    q_v = q_v-alpha*grad;
    q = q_v;
    
    dy = b-opA(q,0);
    lam = max(0,norm(dy,2)/nlvl - 1);
    
    v = lam/((lam+1)*c_A)*opA(dy,1) + q;

    % compute z_{n+1}
    tau = 2/(n+3);
    z = tau*v+(1-tau)*x;
end

result = x;

end