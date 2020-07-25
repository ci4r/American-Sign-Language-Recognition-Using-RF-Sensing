function [state,options,optchanged] = myPlot_filterbank_v2(options, state, flag)
  
M = 5;
prf = 3200;
num_class = 5; 
[accuracy, index] = min(state.Score);
K = 441; % arbitrary length
[H,freq] = Generic_filterbank_v3(M,state.Population(index,:),prf,K);
plot(freq, H);
title({['Optimized Filterbank with accuracy of ', num2str(100*(-accuracy),'%3.1f'), '% ', ' for ', int2str(num_class), ' classes']})%,...
    %['Best Individuals: ',int2str(state.Population(index,:))]});
xlabel('Frequency (Hz)')

global Gen_Hist

Gen_Hist(state.Generation+1).Generation = state.Generation;
Gen_Hist(state.Generation+1).Elites = state.Population(index,:);
Gen_Hist(state.Generation+1).Acc =  num2str(100*(-accuracy),'%3.1f');

end

