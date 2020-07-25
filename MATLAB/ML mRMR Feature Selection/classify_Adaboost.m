function [test_acc] = classify_Adaboost(dataset,labels)
        
        %labels=labels(:);
        rng(1) % constant seed for reproducibility
        tallrng(1) 
        c = cvpartition(labels,'KFold',4);
        for i=1:4
                P=c.test(i);
                testInd=find(P==1);
                trainInd=find(P==0);
                
                Mdl = fitcensemble(dataset(:,trainInd)',labels(trainInd),'Method','Subspace','Learners','discriminant');
                Y_predict = predict(Mdl, dataset(:,testInd)');
                acc(i)= round(sum(labels(testInd) == Y_predict') / length(labels(testInd)),3);
        end
        test_acc = 100*mean(acc); % to minimize fitness function

end