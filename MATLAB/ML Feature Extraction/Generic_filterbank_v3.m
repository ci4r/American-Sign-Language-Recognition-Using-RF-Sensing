function [H,f] = Generic_filterbank_v3(M, f_band, fs, K)
    
    
    f_min = -0.5*fs;          % filter coefficients start at this frequency, i.e. prf/2 (Hz)
    f_max = 0.5*fs;     % filter coefficients end at this frequency (Hz)
    f = linspace( f_min, f_max, K ); % frequency range (Hz), size 1xK
    
    rate = K/fs;
    f_scaled = round(f_band*rate+(K/2));
    findzeros = find(f_scaled == 0);
    num_zeros = length(findzeros);
    for i=1:num_zeros
       f_scaled(findzeros(i)) = 1; 
    end
    
    H = zeros(M,K);
    upper_slope = zeros(M,K);
    lower_slope = zeros(M,K);
    
    m = 1;
    for i=1:3:3*M-2
        upper_length = f_scaled(i+1)-f_scaled(i)+1;
        lower_length = f_scaled(i+2)-f_scaled(i+1)+1;
        
        H(m,f_scaled(i):f_scaled(i+1)) = linspace(0,1,upper_length);
        H(m,f_scaled(i+1):f_scaled(i+2)) = linspace(1,0,lower_length);
        
        m = m+1;
    end
end

