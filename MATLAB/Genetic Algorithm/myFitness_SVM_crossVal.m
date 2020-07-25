function [test_acc] = myFitness_SVM_crossVal(f,spects,num_class,J,M,K,fs,num_epochs)

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
[~, labels_int(1,:)] = max(labels);
%% Train SVM

c = cvpartition(labels_int,'KFold',4);

for i=1:4
    P=c.test(i);
    testInd=find(P==1);
    trainInd=find(P==0);

    Mdl = fitcecoc(dataset(:,trainInd),labels_int(:,trainInd), 'ObservationsIn','columns', 'Learners','svm');

%     Y_train = predict(Mdl, dataset(:,trainInd)');
%     acc(i) = round(sum(labels_int(trainInd) == Y_train') / length(labels_int(trainInd)),3); % compare the predicted vs. actual
%     
    Y_predict = predict(Mdl, dataset(:,testInd)');
    acc2(i) = round(sum(labels_int(testInd) == Y_predict') / length(labels_int(testInd)),3); % compare the predicted vs. actual
    
end

% train_acc = -mean(acc); % to minimize fitness function
test_acc = -mean(acc2); % to minimize fitness function

%% Train Random Forest
% num_tree = 100;
% test_acc = -classify_RandForest(num_tree,dataset,labels_int);
%% Confusion Matrix

% figure(3); plotconfusion(labels(:,tr.testInd),testoutputs,'Testing');
% msg1 = strcat('Predictions: ', int2str(Ytestpred2));   % loading message
% msg2 = strcat('True Labels: ', int2str(true_Ytest));
% msg3 = strcat('Training Acc: ',num2str(-100*train_acc,'%3.1f'), '% / Testing Acc: ', num2str(-100*test_acc,'%3.1f'),'%');
msg3 = strcat('Testing Acc: ', num2str(-100*test_acc,'%3.1f'),'%');

% disp(msg1);
% disp(msg2);
disp(msg3);

end