% FOM_PRIMAL_DUAL_CB Chambolle-Pock primal-dual algorithm for QCBP.
% 
%   Primal-dual iteration method to solve quadratically constrained basis 
%   pursuit (QCBP).
%
% INPUT
% =====
%   x0        - initial value for primal variable
%   y0        - initial value for dual variable
%   tau       - prox step size for primal update
%   sigma     - projection step size for dual update
%   num_iters - number of iterations
%   opA       - QCBP fidelity linear operator
%   b         - QCBP fidelity vector
%   nlvl      - noise level
%   eval_obj  - flag to store objective function and constraint 
%               evaluations
%
% OUTPUT
% ======
%   result    - ergodic average of all primal iterates
%   o_values  - objective function evaluations of primal ergodic averages
%   c_values  - constraint evaluations of primal ergodic averages
%
% NOTES
% =====
%   The input opA should be a function handle, with the first argument a
%   compatible input and the second argument a boolean flag to enable using
%   the transpose of the linear operator (when the flag is nonzero).
%
%   The purpose of expressing opA in this way is to allow for efficient
%   implementations (e.g. fast wavelet transform).
%
%   Additionally, this algorithm is implemented to handle complex-valued
%   data, so opA can be say, the discrete Fourier transform.
%
% REFERENCES
% ==========
%   - TO DO ...
%

function [result, o_values, c_values] = fom_primal_dual_cb(x0, y0, tau, sigma, num_iters, opA, b, nlvl, eval_obj)

x = x0;
y = y0;
Xavg = zeros(size(x0));
Yavg = zeros(size(y0));

o_values = cell(num_iters,1);
c_values = cell(num_iters,1);


for j=0:num_iters-1
    q = x-tau.*opA(y,1);
    x_next = max(abs(q)-tau,0).*sign(q);
    y = y + sigma*opA(2*x_next - x,0) - sigma*ball_projection(y/sigma + opA(2*x_next - x,0), b, nlvl);
    Xavg = (j*Xavg + x_next)/(j+1);
    Yavg = (j*Yavg + y)/(j+1);
    
    x = x_next;

    if eval_obj
        o_values{j+1} = norm(Xavg,1);
        c_values{j+1} = norm(opA(Xavg,0)-b,2);
    end
end

result = Xavg;

end


function z_proj = ball_projection(z, c, rad)
% Computes the projection of z onto the l2-ball centered at c with
% radius rad.
dist = norm(z-c,2);

if dist <= rad
    z_proj = z;
else
    z_proj = c + rad*(z-c)/dist;
end

end