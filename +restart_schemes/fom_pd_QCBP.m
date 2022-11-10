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
%   eval_fns  - cell array of function handles to evaluate on each iterate
%               (assign the empty array [] to disable)
%   F         - a function to use as the argmin over the iterates for
%               eval_fns and the final output (see NOTES).
%
% OUTPUT
% ======
%   result    - 1x2 cell, with ergodic average of primal and dual 
%               iterates in the first and second entry, respectively
%   ev_values - eval_fns function evaluations of primal and dual ergodic 
%               averages, where the function will receive a 1x2 cell,
%               structured as the return value 'result'
%
% NOTES
% =====
%   Defining opA
%   ------------
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
%   The output and evaluations
%   --------------------------
%   An intermediate variable 'Xout' is defined to be the argmin of F over 
%   all previous iterates and the current iterate x. Here Xout is 
%   recomputed at each iteration and is used as input to eval_fns. The 
%   output iterate 'result' is the final Xout vector. The variable Yout is
%   similar except we instead take the argmax over the Lagrangian
%   evaluted at Xout and Yavg.
%
% REFERENCES
% ==========
%   - Chapter 7.5 of "Compressive Imaging: Structure, Sampling, Learning", 
%     Adcock & Hansen (2021). doi:10.1017/9781108377447
%

function [result, ev_values] = fom_pd_QCBP_tweaks(x0, y0, tau, sigma, num_iters, opA, b, nlvl, eval_fns, F)

x = x0;
y = y0;
Xavg = zeros(size(x0));
Yavg = zeros(size(y0));
Xout = x0;
Yout = y0;
ev_values = zeros(length(eval_fns),num_iters);

G = @(xx,yy) real(yy(:)'*(opA(xx,0)-b))-nlvl*norm(yy(:));

for j=0:num_iters-1
    q = x-tau.*opA(y,1);
    x_next = max(abs(q)-tau,0).*sign(q);
    y = y + sigma*opA(2*x_next - x,0) - sigma*ball_projection(y/sigma + opA(2*x_next - x,0), b, nlvl);
    Xavg = (j*Xavg + x_next)/(j+1);
    Yavg = (j*Yavg + y)/(j+1);
    
    x = x_next;

    if F({Xavg,[]})<=F({Xout,[]})
        Xout=Xavg;
    end

    if G(Xout,Yavg)>=G(Xout,Yout)
        Yout=Yavg;
    end

    if ~isempty(eval_fns)
        for fidx=1:length(eval_fns)
            ev_values(fidx,j+1) = eval_fns{fidx}({Xout,Yavg});
        end
    end
end

result = {Xout,Yout};

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