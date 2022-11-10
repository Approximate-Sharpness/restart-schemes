% Inverse square-law sampling for the 2D Fourier inverse problem

function hist = sa_inverse_square_law_hist_2d(N,alpha)

freq = [-N/2+1:1:N/2];
hist = zeros(N,N);

for i=1:length(freq)
    for j=1:length(freq)
        hist(i,j) = (max(1, freq(i)^2 + freq(j)^2))^(-alpha);
    end
end

end