% Two-dimensional discrete gradient operator and its transpose.
%
% INPUT
% =====
%   x    - An N1*N2 vector if mode == 1, otherwise an N1*N2*2 vector. For
%          mode == 1, this corresponds to an input image (reshaped to a 
%          vector).
%   mode - Boolean. If mode == 1, the discrete gradient operator is 
%          applied, otherwise the transpose will be used.
%
% OUTPUT
% ======
%   The result of the operator applied to X.
%
function y = op_discrete_gradient_2d(x, mode, N1, N2)

if (~isvector(x))
    error('Input is not a vector')
end

if (mode == 0)

    X = reshape(x,[N1 N2]);
    Y = zeros([size(X) 2]);

    % vertical (axis) gradient
    for j=1:N2
        for i=1:N1-1
            Y(i,j,1) = X(i+1,j)-X(i,j);
        end
        Y(N1,j,1) = X(1,j)-X(N1,j);
    end

    % horizontal (axis) gradient
    for j=1:N1
        for i=1:N2-1
            Y(j,i,2) = X(j,i+1)-X(j,i);
        end
        Y(j,N2,2) = X(j,1)-X(j,N2);
    end

    y = reshape(Y, [N1*N2*2 1]);

else % transpose

    X = reshape(x,[N1 N2 2]);
    Y = zeros([N1 N2]);

    % vertical (axis) gradient transpose
    for j=1:N2
        for i=2:N1
            Y(i,j) = Y(i,j)+X(i-1,j,1)-X(i,j,1);
        end
        Y(1,j) = Y(1,j)+X(N1,j,1)-X(1,j,1);
    end

    % horizontal (axis) gradient transpose
    for j=1:N1
        for i=2:N2
            Y(j,i) = Y(j,i)+X(j,i-1,2)-X(j,i,2);
        end
        Y(j,1) = Y(j,1)+X(j,N2,2)-X(j,1,2);  
    end

    y = reshape(Y, [N1*N2 1]);

end

end