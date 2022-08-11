% Compute 2D Bernoulli sampling probabilities.
%
% Ported code from nestanet/sampling.py in
% > https://github.com/mneyrane/AS-NESTA-net

function probs = sa_bernoulli_sampling_probs_2d(hist,m)

if all(m*hist <= 1)
    probs = m*hist;
else
    constraint = @(t) bernoulli_probs_constraint(t, m, hist);
    C = bisection(constraint,0,1,1000,1e-12);
    terms = (m*C).*hist;
    probs = min(terms,1);
end

end


function value = bernoulli_probs_constraint(t, m, hist)

terms = (t*m).*hist;
probs = min(terms,1);
value = sum(probs,'all')-m;

end


function r = bisection(f,a,b,N,tol)
% MATLAB function for bisection method.
%
% Inputs: 
% f - the function
% a,b - interval endpoints
% N - maximum number of iterations
% tol - desired tolerance
%
% Outputs:
% r - the computed root

% Check that neither endpoint is a root, and if f(a) and f(b) have the same
% sign, produce an error.
if (f(a)==0)
    r=a;
    return;
elseif (f(b) == 0)
    r=b;
    return;
elseif (f(a) * f(b) > 0)
    error( 'f(a) and f(b) do not have opposite signs' );
end

% Perform the bisection method with N iterations, outputting an error if
% the tolerance tol is not met.
for k=1:N
    p=a+(b-a)/2;
    
    if (f(p)==0 || (b-a)/2<tol)
        r=p;
        return;
    elseif (f(p)*f(a)>0)
        a=p;
    else
        b=p;
    end
end

error('the method did not converge');
end