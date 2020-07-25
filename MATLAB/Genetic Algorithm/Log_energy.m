function [S] = Log_energy(X,H)
%LOG_ENERGY Summary of this function goes here
%Input(s):
%   X   Spectrogram
%   H   Filterbank
%Output(s):
%   S   The Log-energy output of each filter
X = double(X);

n = size(X,2); % X is the spectrogram matrix
m = size(H,1);
S = zeros(m,n); % m by n
    for mx=1:m
        for nx=1:n
            
           S(mx,nx) = log( sum(X(:,nx)'.*H(mx,:)) ); 
%            for j=1:size(H,2)
%                 S(mx,nx) = S(mx,nx) + X(j,nx)*H(mx,j); 
%            end
        end
    end  
S(S<0) = 0;
end

