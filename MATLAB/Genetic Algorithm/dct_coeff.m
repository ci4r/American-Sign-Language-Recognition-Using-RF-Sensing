function [c_dct] = dct_coeff(S,J)
    % S is the log-energy output of filters
    % J is the # of the cepstral coefficients
    M = size(S,1);
    c_dct = zeros(size(S,2),J);

    for j=1:J
        
        for n=1:size(S,2)
            
            for m=1:M
               
                c_dct(n,j) = c_dct(n,j) + S(m,n)*cos(j*(m-0.5)*pi/M);
                
            end
            
        end
    
    end

end