clear; clc; close all;

% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\77 GHz\microDoppler No MTI 65x65\Gray\'; 
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\77 GHz\microDoppler No MTI 201x451\Gray\Cropped\';
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\77 GHz\microDoppler After MTI 65x65\Gray\';
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\77 GHz\microDoppler orig spect 65x65\Gray\';
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\24 GHz\Front\Gray\';
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\10 GHz\microDoppler\Front\Gray\';
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\10 GHz\microDoppler\Corner\Gray\';
% DATA_DIR = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\10 GHz\microDoppler\Side\Gray\';
%% Select savepath
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\77 GHz\No MTI resize201x451 to 65x65\';
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\77 GHz\No MTI 65x65\';    
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\77 GHz\After MTI 65x65\';
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\77 GHz\Spect size orig 65x65';
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\24 GHz\';
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\10 GHz\Front\';
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\10 GHz\Corner\';
% savepath = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\10 GHz\Side\';
%% Native
masterpath = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Native\';
savemaster = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\';
data = {[masterpath, '10 GHz\microDoppler\Front\Gray\'], [masterpath, '10 GHz\microDoppler\Corner\Gray\'],...
    [masterpath, '10 GHz\microDoppler\Side\Gray\'], [masterpath, '77 GHz\microDoppler orig spect 65x65\Gray\'],...
    [masterpath, '24 GHz\Front\No MTI 201x451\Gray\'],[masterpath, '10 GHz\microDoppler\Front\No MTI 201x451\Gray\'],...
    [masterpath, '10 GHz\microDoppler\Corner\No MTI 201x451\Gray\'], [masterpath, '10 GHz\microDoppler\Side\No MTI 201x451\Gray\']};
savedir = {[savemaster, '10 GHz\Front\'], [savemaster, '10 GHz\Corner\'], [savemaster, '10 GHz\Side\'], ...
    [savemaster, '77 GHz\Spect size orig 65x65\'], [savemaster, '24 GHz\No MTI 201x451\'], [savemaster, '10 GHz\Front\No MTI 201x451\'],...
    [savemaster, '10 GHz\Corner\No MTI\'], [savemaster, '10 GHz\Side\No MTI\']};
%% Imitation
% masterpath = 'C:\Users\ekurtoglu\Desktop\ASL All Outputs\Imitation\';
% savemaster = 'C:\Users\mrahman17\Desktop\Genetic Algorithm\Results\ASL\';
% data = {[masterpath, '77 GHz\Front No MTI 201x451\Gray\Cropped\']};
% savedir = {[savemaster, '77 GHz\Imitation\']};
%% Read Data
num_classes = 20;%[5 10 15 20];
accuracy_hist = zeros(length(data), length(num_classes));
for ii=7:length(data)
    DATA_DIR = data{ii};
    savepath = savedir{ii};
    files = dir(DATA_DIR);
    dirFlags = [files.isdir];
    subFolders = files(dirFlags);
    subFolders = subFolders(3:end);
%     for k = 1 : length(subFolders)
%       fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
%     end
    
    for jj=1:length(num_classes)
        num_class = num_classes(jj);%length(subFolders);
        spects = cell(num_class,1);
        message = strcat(['Processing dataset ', int2str(ii), ' of ', int2str(length(data)), ' Class ', int2str(num_classes(jj))]);   % loading message
        disp(message);
        for j=1:num_class
            pattern = strcat(subFolders(j).folder,'\',subFolders(j).name,'\', '*.png');    % file pattern
            files = dir(pattern);
            w = waitbar(0);

            I_MAX = numel(files); % # of files in "files" 

            for i = 1:I_MAX   
                msg = strcat('Processing file ', int2str(i), ' of ', int2str(I_MAX), ' Folder ', int2str(j));   % loading message
                waitbar(i/I_MAX, w, msg);
%                 disp(msg);
                fName = files(i).name;
                spects{j,i} = imresize(rgb2gray(imread(strcat(pattern(1:end-5),fName))),[65 65]); % [height width], [64 120] Gait, [450 100] ASL
            end
            close(w);
        end
        %% Parameters

        num_epochs = 100;
        J = 5; % # of cepstral coefficients
        M = 5; % # of filters
        K = size(spects{1,1},1); % length of each filter
        nvars = 3*M; % start, max, stop point of the filter
        prf = 3200;
        %% Constraints
        b = zeros(20,1);
        A = zeros(20,nvars);

        for i = 0:4
            A(4*i+1, 3*i + 1) = 1;
            A(4*i+1, 3*i + 2) = -1;
            A(4*i+2, 3*i + 2) = 1;
            A(4*i+2, 3*i + 3) = -1;
            % Upper & Lower Bounds
            A(4*i+3, 3*i + 1) = -1;
            A(4*i+4, 3*i + 3) = 1;
            b(4*i+3:4*i+4, 1) = prf/2-1;
        end
        %% Run Genetic Algorithm
        InitPop = [linspace(-prf/5,prf/5,nvars); linspace(-prf/6,prf/6,nvars); linspace(-prf/7,prf/7,nvars)];
        FitFnc = @(f)(myFitness_SVM_crossVal(f,spects,num_class,J,M,K,prf,num_epochs));
        clear global
        global Gen_Hist
        options = optimoptions(@ga,'PlotFcn',{@gaplotbestf,@myPlotGA},...%@myPlot_filterbank_v2},...% 'OutputFcn',{@myOut_history},...
            'Display','iter','MaxGenerations',30, 'FitnessLimit', -1, 'PopulationSize',128);%, 'InitialPopulationMatrix',InitPop);
        tic
        [f,fval] = ga(FitFnc,nvars,A,b,[],[],[],[],[],options); % return optimal values for fbank and optimum value
        elapsedTime = toc;

        accuracy = 0;
        index = 0;
        for i=1:length(Gen_Hist)
            if str2double(Gen_Hist(i).Acc) >= accuracy
                accuracy = str2double(Gen_Hist(i).Acc);
                index = i;
            end
        end
        best_param = Gen_Hist(index).Elites;

        [H,freq] = Generic_filterbank_v3(M,best_param,prf,K);

        figure;
        plot(freq, H);
        title(['Optimized Filterbank with accuracy of ', num2str(accuracy,'%3.1f'), '% ', ' for ', int2str(num_class), ' classes']);
        xlabel(['Frequency (Hz) / Elapsed Time = ', int2str(elapsedTime/3600), ...
            ' hours ', int2str(mod(elapsedTime/60,60)), ' minutes '])
        savename = strcat(savepath,int2str(num_class),'_class_','filterbank_param.mat');
        save(savename,'best_param','accuracy');
        accuracy_hist(ii,jj) = accuracy;
        
    end
end


