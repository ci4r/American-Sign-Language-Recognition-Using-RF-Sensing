clear; clc; close all;

load('8ch_7env_500dct_325fwcc_100lpc_channelwise.mat');
classifier1 = @(dataset,labels)(seq_backward_cross_val(dataset,labels));
classifier2 = @(num_tree,dataset,labels)(classify_RandForest(num_tree,dataset,labels));
classifier3 = @(dataset,labels)(classify_Adaboost(dataset,labels));
classifier4 = @(dataset,labels)(classify_knn(dataset,labels));
num_tree = 100;
num_class = 2:20;
channel = 8;
num_feats = [50 100 120 200];
final_num_feats = num_feats;

ch_acc = zeros(length(num_class),length(num_feats));
ch_feat = cell(length(num_class), length(num_feats));
ch_data = cell(length(num_class), length(num_feats));

final_acc = zeros(1,length(num_class));
final_feat = cell(1,length(num_feats));
final_data = cell(1,length(num_feats));

for j = 1:length(num_class)
        
        for i=1:channel
                temp{i,1} = separate_data{i,1}(:,1:num_class(j)*8);
        end
        separate_data2 = temp;
        sepdata3(1:4) = separate_data2(1:4);
        sepdata3(5:6) = separate_data2(7:8);
        
        labels2 = labels(1:num_class(j)*8);
%         for i = 1:channel
                
%                 dataset = sepdata3{i,1};
                
%                 feats = mrmr_miq_d(dataset',labels2',num_feats(1));                
%                 sel_dataset1 = dataset(feats,:);
%                 test_acc1 = classifier1(sel_dataset1,labels2);
%                 ch_data1{j,i} = sel_dataset1;
%                 
%                 feats = mrmr_miq_d(dataset',labels2',num_feats(2));                
%                 sel_dataset2 = dataset(feats,:);
%                 test_acc2 = classifier2(num_tree,sel_dataset2,labels2);
%                 ch_data2{j,i} = sel_dataset2;
%                 
%                 feats = mrmr_miq_d(dataset',labels2',num_feats(3));                
%                 sel_dataset3 = dataset(feats,:);
%                 test_acc3 = classifier3(sel_dataset3,labels2);
%                 ch_data3{j,i} = sel_dataset3;
%                 
%                 feats = mrmr_miq_d(dataset',labels2',num_feats(4));                
%                 sel_dataset4 = dataset(feats,:);
%                 test_acc4 = classifier4(sel_dataset4,labels2);
%                 ch_data4{j,i} = sel_dataset4;
                
%                 if i==channel
%                         dataset1 = vertcat(ch_data1{j,:});
                        dataset1 = vertcat(sepdata3{:,:});
                        
                        feats = mrmr_miq_d(dataset1',labels2',final_num_feats(1));
                        sel_dataset1 = dataset1(feats,:);
                        test_acc = classifier1(sel_dataset1,labels2);
                        ch_acc(j,1) = test_acc;
                        msg = strcat(['Final accuracy of SVM when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
                        disp(msg);
                        
%                         dataset2 = vertcat(ch_data2{j,:});
                        dataset2 = vertcat(sepdata3{:,:});
                        feats = mrmr_miq_d(dataset2',labels2',final_num_feats(2));
                        sel_dataset2 = dataset2(feats,:);
                        test_acc = classifier2(num_tree,sel_dataset2,labels2);
                        ch_acc(j,2) = test_acc;
                        msg = strcat(['Final accuracy of Random Forest when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
                        disp(msg);
                        
%                         dataset3 = vertcat(ch_data3{j,:});
                        dataset3 = vertcat(sepdata3{:,:});
                        feats = mrmr_miq_d(dataset3',labels2',final_num_feats(3));
                        sel_dataset3 = dataset3(feats,:);
                        test_acc = classifier3(sel_dataset3,labels2);
                        ch_acc(j,3) = test_acc;
                        msg = strcat(['Final accuracy of Adaboost (decision trees) when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
                        disp(msg);
                        
%                         dataset4 = vertcat(ch_data4{j,:});
                        dataset4 = vertcat(sepdata3{:,:});
                        feats = mrmr_miq_d(dataset4',labels2',final_num_feats(4));
                        sel_dataset4 = dataset4(feats,:);
                        test_acc = classifier4(sel_dataset4,labels2);
                        ch_acc(j,4) = test_acc;
                        msg = strcat(['Final accuracy of kNN when # of classes is ',num2str(num_class(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
                        disp(msg);
%                 end
%         end
end
save('classifierVSclass.mat')
figure(1);
hold on
grid on
xlabel('Number of classes');
ylabel('Testing Accuracy (%)');
% title('Performance of Classifiers According to Number of Classes');
set(gca, 'Fontsize', 24, 'Fontweight', 'bold');
p1 = plot(num_class, ch_acc(:,1),'r--*', 'LineWidth',4,'markersize',9);
p2 = plot(num_class, ch_acc(:,2),'b-s', 'LineWidth',6,'markersize',9);
p3 = plot(num_class, ch_acc(:,3),'k-.p', 'LineWidth',4,'markersize',9);
p4 = plot(num_class, ch_acc(:,4),'m:x', 'LineWidth',4,'markersize',9);
methods = {'SVM','Random Forest','Linear Discriminant Analysis','k-NN'};
legend([p1, p2, p3, p4],{methods{1}, methods{2}, methods{3}, methods{4}});
set(gcf, 'color','w');