clear; clc; close all;

% load('Results_MTI0_500DCT.mat');
% mti0 = vertcat(separate_data{:,1});
% load('Results_MTI1_500DCT.mat');
% mti1 = vertcat(separate_data{:,1});
load('Results_MTI2_500DCT.mat');
mti2 = vertcat(separate_data{:,1});
clearvars -except mti0 mti1 mti2 labels

classifier1 = @(dataset,labels)(seq_backward_cross_val(dataset,labels));
classifier2 = @(num_tree,dataset,labels)(classify_RandForest(num_tree,dataset,labels));
classifier3 = @(dataset,labels)(classify_Adaboost(dataset,labels));
classifier4 = @(dataset,labels)(classify_knn(dataset,labels));

num_tree = 100;
num_class = 2:20;
num_feats = [120 150 150];
%% Find Best Performing Classes
% feats0 = mrmr_miq_d(mti0',labels',num_feats(1));
% feats1 = mrmr_miq_d(mti1',labels',num_feats(2));
feats2 = mrmr_miq_d(mti2',labels',num_feats(3));

% [test_acc0, y0, yhat0] = classifier(num_tree,mti0(feats0,:),labels);
% [test_acc1, y1, yhat1] = classifier(num_tree,mti1(feats1,:),labels);
[test_acc2, y2, yhat2] = classifier2(num_tree,mti2(feats2,:),labels);

% conf0 = confusionmat(y0, yhat0);
% scores0 = diag(conf0);
% conf1 = confusionmat(y1, yhat1);
% scores1 = diag(conf1);
conf2 = confusionmat(y2, yhat2);
scores2 = diag(conf2);

% [B0, I0] = sort(scores0,'descend');
% [B1, I1] = sort(scores1,'descend');
[B2, I2] = sort(scores2,'descend');
%% Accuracy vs # of classes
final_acc = zeros(4,length(num_class));

for j = 1:length(num_class)
    clear labels2 sel_dataset0 sel_dataset1 sel_dataset2
    for i = 1:j+1
        labels2((i-1)*8+1:i*8) = labels((I2(i)-1)*8+1:I2(i)*8);
%         sel_dataset0(:,(i-1)*8+1:i*8) = mti0(feats0,(I0(i)-1)*8+1:I0(i)*8);
%         sel_dataset1(:,(i-1)*8+1:i*8) = mti1(feats1,(I1(i)-1)*8+1:I1(i)*8);
        sel_dataset2(:,(i-1)*8+1:i*8) = mti2(feats2,(I2(i)-1)*8+1:I2(i)*8);
    end
    
    if j == length(num_class)
%         sel_dataset0 = mti0(feats0,:);
%         sel_dataset1 = mti1(feats1,:);
        sel_dataset2 = mti2(feats2,:);
        labels2 = labels;
    end
    
    test_acc = classifier1(sel_dataset2,labels2);
    final_acc(1,j) = test_acc;
    msg = strcat(['Final accuracy of SVM when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
    disp(msg);

    [test_acc, y1, yhat1] = classifier2(num_tree,sel_dataset2,labels2);
    final_acc(2,j) = test_acc;
    msg = strcat(['Final accuracy of Random Forest when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
    disp(msg);

    test_acc2 = classifier3(sel_dataset2,labels2);
    final_acc(3,j) = test_acc2;
    msg = strcat(['Final accuracy of LDA when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc2,'%3.1f'),'%.']);
    disp(msg);
    
    test_acc3 = classifier4(sel_dataset2,labels2);
    final_acc(4,j) = test_acc3;
    msg = strcat(['Final accuracy of k-NN when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc3,'%3.1f'),'%.']);
    disp(msg);
end
% mod0 = sort(mod(feats0,932));
% mod1 = sort(mod(feats1,932));
% mod2 = sort(mod(feats2,932));
% 
% mtis = sort(mod(feats2(feats2<=4660),932));
% nomtis = sort(mod(feats2(feats2>4660),932));
% save('MTIvsNoMTI.mat')
figure(1);
hold on
grid on
xlabel('Number of classes');
ylabel('Testing Accuracy (%)');
% title('Performance of Classifiers According to Number of Classes');
set(gca, 'Fontsize', 24, 'Fontweight', 'bold');
p1 = plot(num_class, final_acc(1,:),'r--*', 'LineWidth',4,'markersize',9);
p2 = plot(num_class, final_acc(2,:),'b-s', 'LineWidth',4,'markersize',9);
p3 = plot(num_class, final_acc(3,:),'k-.p', 'LineWidth',4,'markersize',9);
p4 = plot(num_class, final_acc(4,:),'m:x', 'LineWidth',6,'markersize',9);
methods = {'SVM','Random Forest','Linear Discriminant Analysis','k-NN'};
legend([p1, p2, p3, p4],{methods{1}, methods{2}, methods{3}, methods{4}}, 'location','northeast');
set(gcf, 'color','w');