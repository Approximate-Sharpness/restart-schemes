% Gradient of the l1-norm's Moreau envelope. 
%
% Used in 'fom_nesta.m'.

function out = op_smooth_l1_gradient(x, mu)
    
out = zeros(size(x));
for i=1:length(x)
    if (abs(x(i)) <= mu)
        out(i) = x(i)/mu;
    else
        out(i) = x(i)/abs(x(i));
    end
end