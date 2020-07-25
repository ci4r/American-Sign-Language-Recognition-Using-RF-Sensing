function state = myPlotGA(options, state, flag)

[acc, index] = min(state.Score); % Best score in the current generation
current_accuracy = -100*acc;

global Gen_Hist

Gen_Hist(state.Generation+1).Generation = state.Generation;
Gen_Hist(state.Generation+1).Elites = state.Population(index,:);
Gen_Hist(state.Generation+1).Acc =  num2str(current_accuracy,'%3.1f');

best_accuracy = 0;
% index = 0;
for i=1:length(Gen_Hist)
    if str2double(Gen_Hist(i).Acc) >= best_accuracy
        best_accuracy = str2double(Gen_Hist(i).Acc);
%         index = i;
    end
end

if(strcmp(flag,'init')) % Set up the plot
    xlim([1,options.MaxGenerations]);
    hold on;
    xlabel Generation
    ylabel('Accuracy of Elite Individuals')
    title(['Testing Accuracy: ',num2str(best_accuracy,'%3.1f'),'%'])
end
plot(state.Generation,current_accuracy,'xr','LineWidth',2)
title(['Best Testing Accuracy: ',num2str(best_accuracy,'%3.1f'),'%'])
end