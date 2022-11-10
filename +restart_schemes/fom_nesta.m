% FOM_NESTA NESTerov's Algorithm to solve analysis-QCBP.
% 
%   An accelerated projected gradient descent algorithm to solve
%   analysis quadratically constrained basis pursuit (A-QCBP).
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
%   F         - a function to use as the argmin over the iterates for
%               eval_fns and the final output (see NOTES).
%
% OUTPUT
% ======
%   result    - approximate minimizer of aQCBP
%   ev_values - eval_fns function evaluations of iterates
%
% NOTES
% =====
%   
%   Defining opA and opW
%   --------------------
%   To use NESTA correctly, the linear operator A must satisfy A*A' = c.*I 
%   where I is the identity map and c > 0 is a constant. In other words,
%   representing A as a matrix, its rows are mutually orthonormal up to
%   a constant factor.
%
%   opA and opW should be function handles representing linear operators,
%   each with their first argument a compatible input and their second 
%   argument a boolean flag to enable using the adjoint of the linear 
%   operator when the flag is nonzero (i.e. true).
%
%   The purpose of expressing opA and opW in this way is to allow for 
%   efficient implementations (e.g. fast wavelet transform).
%
%   Additionally, this algorithm is implemented to handle complex-valued
%   data, so opA can be e.g. the discrete Fourier transform, and the
%   iterates are now (possibly) complex-valued.
%
%   The output and evaluations
%   --------------------------
%   An intermediate variable 'xout' is defined to be the argmin of F over 
%   all previous iterates and the current iterate x. Here xout is 
%   recomputed at each iteration and is used as input to eval_fns. The 
%   output iterate 'result' is the final xout vector.
%
%
% REFERENCES
% ==========
%   - "NESTA: A Fast and Accurate First-Order Method for Sparse Recovery",
%     Becker, et al (2011). doi:10.1137/090756855
%

function [result, ev_values] = fom_nesta(z0, opA, c_A, b, opW, L_W, num_iters, nlvl, mu, eval_fns, F)

z = z0;
q_v = z0;

xout = z0;

F_min = F(xout);

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
    
    % argmin over inner iterations
    F_eval_x = F(x);
    if F_eval_x < F_min
        xout = x;
        F_min = F_eval_x;
    end
    
    if ~isempty(eval_fns)
        for fidx=1:length(eval_fns)
            ev_values(fidx,n+1) = eval_fns{fidx}(xout);
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

result = xout;

end