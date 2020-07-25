function [test_acc] = myFitness_RandForest(f,spects,num_class,J,M,K,fs,num_epochs)

sample = 1;

[H, freq] = Generic_filterbank_v3(M,f,fs,K); % create a filterbank

%% Read Data Matrices and Extract DCT Coeffs.
num_samples = sum(~cellfun('isempty',spects(1,:)));

% labels = zeros(num_class,num_class*num_samples);
% dataset = zeros(J*size(spects{1,1},2),num_class*num_samples);
    
for j=1:num_class
    for i = 1:size(spects,2)
        if ~isempty(spects{j,i})
%             msg = strcat('Processing file ', int2str(i), ' of ', int2str(size(spects,2) ), ' Folder ', int2str(j));   % loading message
%             disp(msg);
            S = Log_energy(spects{j,i},H); % calculate log-energy output
            C_DCT2 = dct_coeff(S,J); % cepstral coefficients
            labels(j,sample) = 1;
            dataset(:,sample) = reshape(C_DCT2,1,J*size(C_DCT2,1));
            sample = sample + 1;
        end   
    end
end
%% Create Table
[~, int_labels] = max(labels);
Tbl = array2table(dataset');
NumTrees = 600;
trainRatio = 0.7;
valRatio = 0;
testRatio = 0.3;
Q = size(Tbl,1);
[trainInd,valInd,testInd] = dividerand(Q,trainRatio,valRatio,testRatio);
%% Train Random Forest

Mdl = TreeBagger(NumTrees, Tbl(trainInd,:), int_labels(trainInd));

Y_train = str2num(cell2mat(predict(Mdl, Tbl(trainInd,:))));
acc = round(sum(int_labels(trainInd) == Y_train') / length(int_labels(trainInd)),3); % compare the predicted vs. actual
train_acc = -acc; % to minimize fitness function

Y_predict = str2num(cell2mat(predict(Mdl, Tbl(testInd,:))));
acc = round(sum(int_labels(testInd) == Y_predict') / length(int_labels(testInd)),3); % compare the predicted vs. actual
test_acc = -acc; % to minimize fitness function
%% Confusion Matrix

% figure(3); plotconfusion(labels(:,tr.testInd),testoutputs,'Testing');
% msg1 = strcat('Predictions: ', int2str(Ytestpred2));   % loading message
% msg2 = strcat('True Labels: ', int2str(true_Ytest));
msg3 = strcat('Training Acc: ',num2str(-100*train_acc,'%3.1f'), '% / Testing Acc: ', num2str(-100*test_acc,'%3.1f'),'%');
% disp(msg1);
% disp(msg2);
disp(msg3);

end