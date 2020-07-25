clear; clc; close all;

load('Transformed_77_7env_500dct_325fwcc_100lpc_channelwise.mat');
% labels = labels_int_hist{1,1};
% classifier = @(num_tree,dataset,labels)(classify_RandForest(num_tree,dataset,labels));
classifier = @(num_tree,xtrain,ytrain,xtest,ytest)(classify_RF_tr_tst(num_tree,xtrain,ytrain,xtest,ytest));
% classifier = @(dataset,labels)(classify_Adaboost(dataset,labels));
%classifier = @(dataset,labels)(classify_knn(dataset,labels));
num_tree = 100;
num_class = 20;
channel = 1;
num_feats = [20 50 100 120 150 190 200 250 500 700 1000];
% num_feats = [100];
final_num_feats = num_feats;

% for i=1:length(separate_data)
%         temp{i,1} = separate_data{i,1}(:,1:num_class*8);
% end
% separate_data = temp;
% labels = labels(1:num_class*8);
ch_acc = zeros(length(num_feats),channel);
ch_feat = cell(length(num_feats), channel);
ch_data = cell(length(num_feats), channel);

final_acc = zeros(1,length(num_feats));
final_feat = cell(1,length(num_feats));
final_data = cell(1,length(num_feats));

% fifthch = 7;
% sel_data(1:4) = separate_data(1:4);
% sel_data(5) = separate_data(fifthch);
% sel_data(6) = separate_data(8);
% sel_data = separate_data';
trainX = separate_data{1,1};
trainY = labels;
load('New_Native_77_7env_500dct_325fwcc_100lpc_channelwise.mat');
testX = separate_data{1,1};
testY = labels;
for j = 1:length(num_feats)
        for i = 1:channel
                
                dataset = trainX;
                feats = mrmr_miq_d(dataset',trainY',num_feats(j));
                ch_feat{j,i} = feats;
                
                sel_dataset = dataset(feats,:);
                ch_data{j,i} = sel_dataset;
                %test_acc = seq_backward_cross_val(sel_dataset,labels);
                test_acc = classifier(num_tree,sel_dataset,trainY,testX(feats,:),testY);
                ch_acc(j,i) = test_acc;
                
                msg = strcat(['Accuracy of Channel #',num2str(i),' with ',num2str(num_feats(j)), ' features: ',num2str(test_acc,'%3.1f'),'%.']);
                disp(msg);
                
%                 if i==channel
%                         dataset = vertcat(ch_data{j,:});
%                         feats = mrmr_miq_d(dataset',labels',final_num_feats(j));
%                         final_feat{1,j} = feats;
%                         
%                         sel_dataset = dataset(feats,:);
%                         final_data{1,j} = sel_dataset;
%                         %test_acc = seq_backward_cross_val(sel_dataset,labels);
%                         test_acc = classifier(num_tree,sel_dataset,labels);
%                         final_acc(1,j) = test_acc;
%                         
%                         msg = strcat(['Final accuracy when # of features is ',num2str(num_feats(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
%                         disp(msg);
%                 end
        end
end
save('results_transformed.mat')
%% Combine Best Channels (Blue Cells in Excel)
%load('best_channels.mat');

num_feats = [120 120 50 120 20 20 100 120]; % 20 100 190
final_features = [50 100 120 150 190 200 250];
ch_feat = cell(1,channel);

% sel_data(~cellfun('isempty',sel_data'));
for i=1:8%channel
        tmp = separate_data{i,1};
        ch_feat{i} = mrmr_miq_d(tmp',labels',num_feats(i));
        tmpdataset{i} = separate_data{i,1}(ch_feat{i},:);
        
end
fifthch = 7;
sel_data(1:4) = tmpdataset(1:4);
sel_data(5) = tmpdataset(fifthch);
idx = 1:4;
idx(5) = fifthch;

ch_feat_stack = horzcat(ch_feat{1,idx});

final_feat = cell(1,length(final_features));
dataset = vertcat(sel_data{:,:});
for j=1:length(final_features)

                        
                        feats = mrmr_miq_d(dataset',labels',final_features(j));
                        final_feat{1,j} = feats;
                        
                        sel_dataset = dataset(feats,:);
                        final_data{1,j} = sel_dataset;
                        %test_acc = seq_backward_cross_val(sel_dataset,labels);
                        test_acc = classifier(num_tree,sel_dataset,labels);
                        final_acc(1,j) = test_acc;
                        
                        msg = strcat(['Final accuracy when # of features is ',num2str(final_features(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
                        disp(msg);
end

selected_feats = sort(ch_feat_stack(feats));
%save('final_feat.mat','final_feat');
%% Stack all features and channels directly (Last Table in excel)

all_data = vertcat(separate_data{:,1});

fifthch = [6];
all_data2(1:4) = separate_data(1:4);
all_data2(5) = separate_data(5);
for i = 1:length(fifthch)
all_data2(6) = separate_data(fifthch(i));
all_data = vertcat(all_data2{:,:});
all_data = vertcat(separate_data{:,:});
% for i=1:7
%         all_data2{i,1} = separate_data{i,1}(8:507,:);
% end
% all_data = vertcat(all_data2{:,1});
final_feats = final_features;
feat_list = repmat([1:932],1,5);
for j=1:length(final_feats)

                        dataset = all_data;
                        feats = mrmr_miq_d(dataset',labels',final_feats(j));
                        final_featss{i,j} = feats;
                        
                        sel_dataset = dataset(feats,:);
                        final_datass{i,j} = sel_dataset;
                        test_acc = classifier(num_tree,sel_dataset,labels);
                        final_accss(i,j) = test_acc;
                        
                        msg = strcat([' CH: ',num2str(fifthch(i)),', Final accuracy when # of features is ',num2str(final_feats(j)), ': ',num2str(test_acc,'%3.1f'),'%.']);
                        disp(msg);
end
% final_feat = sort(feat_list(feats));
end