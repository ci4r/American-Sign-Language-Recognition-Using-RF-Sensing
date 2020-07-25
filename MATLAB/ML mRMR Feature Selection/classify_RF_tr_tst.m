function [test_acc] = classify_RF_tr_tst(num_tree,xtrain,ytrain,xtest,ytest)
        
%         rng(1) % constant seed for reproducibility
%         tallrng(1) 
        for i=1:4
                
                Mdl = TreeBagger(num_tree,xtrain',ytrain);
                Y_predict2 = predict(Mdl, xtest');
                for j=1:length(Y_predict2)
                        Y_predict(j) = str2num(Y_predict2{j,1});
                end
                acc(i)= round(sum(ytest == Y_predict) / length(ytest),3);
                clear Y_predict
        end
        test_acc = 100*mean(acc); % to minimize fitness function

end
