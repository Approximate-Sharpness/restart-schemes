% Two-dimensional Fourier operator with subsampling. Here the operator is
% normalized so that the corresponding Fourier transform is unitary.
%
% INPUT
% =====
%   x    - Vector.
%   mode - Boolean. If mode == 1, the subsampled Fourier transform is 
%          applied, otherwise the conjugate transpose will be used
%   N    - Size of the vector x.
%   idx  - The matrix indices one would like to sample, given in an linear
%          order (see sub2ind(..) for conversion to linear order)
%
% OUTPUT
% ======
%   The result of the operator applied to x.  
%
function y = op_fourier_1d(x, mode, N, idx)

if (~isvector(x))
    error('Input is not a vector');
end

if (mode == 0) 

    z = fftshift(fft(x))/sqrt(N);
    y = z(idx);
    y = y(:);

else % adjoint

    z = zeros([N, 1]);
    z(idx) = x;
    y = ifft(ifftshift(z))*sqrt(N);
    
end

end