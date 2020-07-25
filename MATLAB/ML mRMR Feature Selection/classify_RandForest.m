function [test_acc] = classify_RandForest(num_tree,dataset,labels)
        
        %labels=labels(:);
        rng(1) % constant seed for reproducibility
        tallrng(1) 
        c = cvpartition(labels,'KFold',4);
        for i=1:4
                P=c.test(i);
                testInd=find(P==1);
                trainInd=find(P==0);
                
                Mdl = TreeBagger(num_tree,dataset(:,trainInd)',labels(trainInd));
                Y_predict2 = predict(Mdl, dataset(:,testInd)');
                for j=1:length(Y_predict2)
                        Y_predict(j) = str2num(Y_predict2{j,1});
                end
                acc(i)= round(sum(labels(testInd) == Y_predict) / length(labels(testInd)),3);
                clear Y_predict
        end
        test_acc = 100*mean(acc); % to minimize fitness function

end
