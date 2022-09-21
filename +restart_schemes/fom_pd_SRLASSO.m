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
%   opA       - SR-LASSO fidelity term linear operator
%   b         - SR-LASSO fidelity term vector
%   lambda    - weight on 1-norm term in SR-LASSO
%   num_iters - number of iterations
%   eval_fns  - cell array of function handles to evaluate on each iterate
%               (assign the empty array [] to disable)
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

function [result, ev_values] = fom_pd_SRLASSO(x0, y0, tau, sigma, opA, b, lambda, num_iters, eval_fns)

F = @(xx) lambda*norm(xx,1) + norm(opA(xx,0)-b,2);
%G = @(xx,yy) real(yy(:)'*(opA(xx,0)-b))-nlvl*norm(yy(:));

x = x0;
y = y0;
Xavg = zeros(size(x0));
Yavg = zeros(size(y0));
Xout = x0;
Yout = y0;
ev_values = zeros(length(eval_fns),num_iters);

for j=0:num_iters-1
    xshift = x-tau.*opA(y,1);
    x_next = max(abs(xshift)-tau*lambda,0).*sign(xshift);
    yshift = y + sigma.*opA(2*x_next - x,0) - sigma.*b;
    y = min(1,1/norm(yshift,2)).*yshift;
    Xavg = (j.*Xavg + x_next)/(j+1);
    Yavg = (j.*Yavg + y)/(j+1);
    
    x = x_next;
    
    % select Xout from x or Xavg, by argmin with respect to F
    if F(x) < F(Xout)
        Xout=x;
    end
    if F(Xavg) < F(Xout)
        Xout=Xavg;
    end
    % select Yout from y or Yavg, by argmin with respect to G and Xout
    %if G(Xout,y) > G(Xout,Yout)
    %    Yout=y;
    %end
    %if G(Xout,Yavg) > G(Xout,Yout)
    %    Yout=Yavg;
    %end
    
    % evaluate Xout and Yout at the user-defined cell of functions
    if ~isempty(eval_fns)
        for fidx=1:length(eval_fns)
            ev_values(fidx,j+1) = eval_fns{fidx}({Xout,Yout});
        end
    end
end

result = {Xout,Yout};

end