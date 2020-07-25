function [test_acc] = classify_RF_im(num_tree,xtrain,ytrain,xtest,ytest)
        
        %labels=labels(:);
        rng(1) % constant seed for reproducibility
        tallrng(1) 
        
                
        Mdl = TreeBagger(num_tree,xtrain',ytrain);
        Y_predict2 = predict(Mdl, xtest');
        for j=1:length(Y_predict2)
                Y_predict(j) = str2num(Y_predict2{j,1});
        end
        acc = round(sum(ytest == Y_predict) / length(ytest),3);
       
        test_acc = 100*acc; % to minimize fitness function

end