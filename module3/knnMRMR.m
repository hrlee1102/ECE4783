
clear;
clc;
format long
%% Load Dataset
Xtrain = load('Xtrain.mat').Xtrain;
Ytrain = load('Ytrain.mat').Ytrain;
Xtest = load('Xtest.mat').Xtest;
Ytest = load('Ytest.mat').Ytest;
cvKF5 = load('cvKF5.mat','cvKF5').cvKF5;
cvKF10 = load('cvKF10.mat','cvKF10').cvKF10;
cvKF20 = load('cvKF20.mat','cvKF20').cvKF20;
%% Load MRMR
MRMR = load('MRMR.mat').MRMR;
idxMRMR = MRMR(1,:);
scores = MRMR(2,:);
%% K vs Num. Features (K-Fold = 5)
nbMRMR_5 = zeros(12,1);
AUC_5 = zeros(12,1);
F1_5 = zeros(12,1);

for f=1:12
    
    nbMRMR_CV = fitcnb(Xtrain(:,idxMRMR(1:f)),Ytrain,'CVPartition',cvKF5);
    [nbMRMR_pred, scores] = kfoldPredict(nbMRMR_CV);
    nbMRMR_5(f) = mean(Ytrain == nbMRMR_pred);
        
    [~,~,~,AUCnb_n] = perfcurve(Ytrain,scores(:,1),'Necrosis');
    [~,~,~,AUCnb_s] = perfcurve(Ytrain,scores(:,2),'Stroma');
    [~,~,~,AUCnb_t] = perfcurve(Ytrain,scores(:,3),'Tumor');
    AUC_5(f) = (AUCnb_n + AUCnb_s + AUCnb_t)/3;
    tpNk = sum(ismember(Ytrain,'Necrosis') & ismember(nbMRMR_pred,'Necrosis'));
        tpSk = sum(ismember(Ytrain,'Stroma') & ismember(nbMRMR_pred,'Stroma'));
        tpTk = sum(ismember(Ytrain,'Tumor') & ismember(nbMRMR_pred,'Tumor'));
        fpNk = sum(ismember(nbMRMR_pred,'Necrosis')) - tpNk;
        fpSk = sum(ismember(nbMRMR_pred,'Stroma')) - tpSk;
        fpTk = sum(ismember(nbMRMR_pred,'Tumor')) - tpTk;
        fnNk = sum(ismember(Ytrain, 'Necrosis')) - tpNk;
        fnSk = sum(ismember(Ytrain, 'Stroma')) - tpSk;
        fnTk = sum(ismember(Ytrain, 'Tumor')) - tpTk;
        f1Nk = tpNk/(tpNk + (fpNk+fnNk)/2);
        f1Sk = tpSk/(tpSk + (fpSk+fnSk)/2);
        f1Tk = tpTk/(tpTk + (fpTk+fnTk)/2);
        F1_5(f) = (f1Nk + f1Sk + f1Tk)/3;

        
end
%% Heatmap Accuracy
% xvals_numF = num2cell(Ks);
% yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(nbMRMR_5,'Colormap', jet, 'CellLabelColor','none');
title('Naive Bayes: Accruacy 5-Fold');
% h5.XLabel = 'K_K_N_N';
% h5.YLabel = 'Num of Features';

%% Heatmap AUC
figure;
h = heatmap(AUC_5,'Colormap', jet, 'CellLabelColor','none');
title('Naive Bayes: AUC 5-Fold');
% h.XLabel = 'K_K_N_N';
% h.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(F1_5,'Colormap', jet, 'CellLabelColor','none');
title('Naive Bayes: F1 Score 5-Fold');
% h.XLabel = 'K_K_N_N';
% h.YLabel = 'Num of Features';
%% K vs Num. Features (K-Fold = 10)
nbMRMR_10 = zeros(12,1);
AUC_10 = zeros(12,1);
F1_10 = zeros(12,1);
for f=1:12
    
    nbMRMR_CV = fitcnb(Xtrain(:,idxMRMR(1:f)),Ytrain,'CVPartition',cvKF10);
    [nbMRMR_pred, scores] = kfoldPredict(nbMRMR_CV);
    nbMRMR_10(f) = mean(Ytrain == nbMRMR_pred);
        
    [~,~,~,AUCnb_n] = perfcurve(Ytrain,scores(:,1),'Necrosis');
    [~,~,~,AUCnb_s] = perfcurve(Ytrain,scores(:,2),'Stroma');
    [~,~,~,AUCnb_t] = perfcurve(Ytrain,scores(:,3),'Tumor');
    AUC_10(f) = (AUCnb_n + AUCnb_s + AUCnb_t)/3;
    tpNk = sum(ismember(Ytrain,'Necrosis') & ismember(nbMRMR_pred,'Necrosis'));
        tpSk = sum(ismember(Ytrain,'Stroma') & ismember(nbMRMR_pred,'Stroma'));
        tpTk = sum(ismember(Ytrain,'Tumor') & ismember(nbMRMR_pred,'Tumor'));
        fpNk = sum(ismember(nbMRMR_pred,'Necrosis')) - tpNk;
        fpSk = sum(ismember(nbMRMR_pred,'Stroma')) - tpSk;
        fpTk = sum(ismember(nbMRMR_pred,'Tumor')) - tpTk;
        fnNk = sum(ismember(Ytrain, 'Necrosis')) - tpNk;
        fnSk = sum(ismember(Ytrain, 'Stroma')) - tpSk;
        fnTk = sum(ismember(Ytrain, 'Tumor')) - tpTk;
        f1Nk = tpNk/(tpNk + (fpNk+fnNk)/2);
        f1Sk = tpSk/(tpSk + (fpSk+fnSk)/2);
        f1Tk = tpTk/(tpTk + (fpTk+fnTk)/2);
        F1_10(f) = (f1Nk + f1Sk + f1Tk)/3;
        
end
%% Heatmap Accuracy
% xvals_numF = num2cell(Ks);
% yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(nbMRMR_10, 'Colormap', jet, 'CellLabelColor','none');
title('Naive Bayes: Accuracy 10-Fold');
% h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
%% Heatmap AUC
figure;
h2 = heatmap(AUC_10, 'Colormap', jet, 'CellLabelColor','none');
title('Naive Bayes: AUC 10-Fold');
% h.XLabel = 'K_K_N_N';
h2.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(F1_10, 'Colormap', jet, 'CellLabelColor','none');
title('Naive Bayes: F1 Score 10-Fold');
% h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% External Validtion
nbMRMR_flat_1 = reshape(nbMRMR_10,[],1);
nbMRMR_flat = sort(unique(nbMRMR_flat_1), 'descend');
fs = [];
nbMRMR_CV_preds = [];
for a=1:length(nbMRMR_flat)
    inds = find(nbMRMR_10 == nbMRMR_flat(a));
    for i = 1:length(inds)
        fs = [fs, inds(i)];
        nbMRMR_CV_preds = [nbMRMR_CV_preds, nbMRMR_flat(a)];
    end
end
%%
nbMRMR_preds = [];
for i=1:length(fs)
    f = fs(i);
    nbMRMR = fitcnb(Xtrain(:,idxMRMR(1:f)),Ytrain);
    pred = predict(nbMRMR, Xtest(:,idxMRMR(1:f)));
    nbMRMR_preds = [nbMRMR_preds, mean(pred==Ytest)]; % accuracy
    
end

%% Plot Acuracy
figure;
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on

scatter(nbMRMR_CV_preds,nbMRMR_preds,'+','Linewidth',1);

xlabel('Internal');
ylabel('External');

title('Internal Validation vs. External Validation');
%%
%12
master_nb = [nbMRMR_preds.', fs.'];
master_nb = sortrows(master_nb,1,'descend');
bf = master_nb(1,2);
nbMRMR = fitcnb(Xtrain(:,idxMRMR(1:bf)),Ytrain);
[pred,score] = predict(nbMRMR, Xtest(:,idxMRMR(1:bf)));
acc = mean(pred==Ytest) * 100 % accuracy
figure;
cm = confusionchart(cellstr(Ytest),pred);
title(['Naive Bayes: ', num2str(acc), '%'])
%%

[X_nb_n,Y_nb_n,T,AUCnb_n] = perfcurve(Ytest,score(:,1),'Necrosis');
[X_nb_s,Y_nb_s,T,AUCnb_s] = perfcurve(Ytest,score(:,2),'Stroma');
[X_nb_t,Y_nb_t,T,AUCnb_t] = perfcurve(Ytest,score(:,3),'Tumor');
%%
figure;
plot(X_knn_n, Y_knn_n,'Linewidth',1.5)
hold on
plot(X_rf_n, Y_rf_n,'Linewidth',1.5)
plot(X_nb_n, Y_nb_n,'Linewidth',1.5)
title('AUC-Necrosis');
legend('KNN','Random Forest','Naive Bayes');



clear
clc
%% Load Dataset
Xtrain = load('Xtrain.mat').Xtrain;
Ytrain = load('Ytrain.mat').Ytrain;
Xtest = load('Xtest.mat').Xtest;
Ytest = load('Ytest.mat').Ytest;
cvKF5 = load('cvKF5.mat','cvKF5').cvKF5;
cvKF10 = load('cvKF10.mat','cvKF10').cvKF10;
cvKF20 = load('cvKF20.mat','cvKF20').cvKF20;
%% Load MRMR
MRMR = load('MRMR.mat').MRMR;
idxMRMR = MRMR(1,:);
scores = MRMR(2,:);
%% Random Forest


%% K vs Num. Features (K-Fold = 5)
Ks =[50:10:150];
rfMRMR_5 = zeros(12,length(Ks));
AUC_5 = zeros(12,length(Ks));
F1_5 = zeros(12,length(Ks));

for f=1:12
    for k=1:length(Ks)
        rfMRMR_CV = fitensemble(Xtrain(:,idxMRMR(1:12)),Ytrain, 'Bag', k, 'Tree', 'Type', 'classification', 'CVPartition', cvKF5);
        [rfMRMR_pred,scores] = kfoldPredict(rfMRMR_CV);
        rfMRMR_5(f,k) = mean(Ytrain == rfMRMR_pred);
        
        [~,~,~,AUCrf_n] = perfcurve(Ytrain,scores(:,1),'Necrosis');
        [~,~,~,AUCrf_s] = perfcurve(Ytrain,scores(:,2),'Stroma');
        [~,~,~,AUCrf_t] = perfcurve(Ytrain,scores(:,3),'Tumor');
        AUC_5(f,k) = (AUCrf_n + AUCrf_s + AUCrf_t)/3;
        tpNk = sum(ismember(Ytrain,'Necrosis') & ismember(rfMRMR_pred,'Necrosis'));
        tpSk = sum(ismember(Ytrain,'Stroma') & ismember(rfMRMR_pred,'Stroma'));
        tpTk = sum(ismember(Ytrain,'Tumor') & ismember(rfMRMR_pred,'Tumor'));
        fpNk = sum(ismember(rfMRMR_pred,'Necrosis')) - tpNk;
        fpSk = sum(ismember(rfMRMR_pred,'Stroma')) - tpSk;
        fpTk = sum(ismember(rfMRMR_pred,'Tumor')) - tpTk;
        fnNk = sum(ismember(Ytrain, 'Necrosis')) - tpNk;
        fnSk = sum(ismember(Ytrain, 'Stroma')) - tpSk;
        fnTk = sum(ismember(Ytrain, 'Tumor')) - tpTk;
        f1Nk = tpNk/(tpNk + (fpNk+fnNk)/2);
        f1Sk = tpSk/(tpSk + (fpSk+fnSk)/2);
        f1Tk = tpTk/(tpTk + (fpTk+fnTk)/2);
        F1_5(f,k) = (f1Nk + f1Sk + f1Tk)/3;

        
    end
end
%% Heatmap Accuracy
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(xvals_numF, yvals_k, rfMRMR_5, 'Colormap', jet, 'CellLabelColor','none');
title('Accruacy 5-Fold');
h5.XLabel = 'K_R_F';
h5.YLabel = 'Num of Features';

%% Heatmap AUC
figure;
h = heatmap(xvals_numF, yvals_k, AUC_5, 'Colormap', jet, 'CellLabelColor','none');
title('AUC 5-Fold');
h.XLabel = 'K_R_F';
h.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(xvals_numF, yvals_k,F1_5, 'Colormap', jet, 'CellLabelColor','none');
title('F1 Score 5-Fold');
h.XLabel = 'K_R_F';
h.YLabel = 'Num of Features';
%%
figure;
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
subplot(3,1,1);
heatmap(xvals_numF, yvals_k, rfMRMR_5,'Colormap', jet, 'CellLabelColor','none');
title('Random Forest: Accruacy 5-Fold');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
subplot(3,1,2);
heatmap(xvals_numF, yvals_k, AUC_5,'Colormap', jet, 'CellLabelColor','none');
title('Random Forest: AUC 5-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
subplot(3,1,3);
heatmap(xvals_numF, yvals_k,F1_5,'Colormap', jet, 'CellLabelColor','none');
title('Random Forest: F1 Score 5-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% K vs Num. Features (K-Fold = 10)
Ks =[50:10:150];
rfMRMR_10 = zeros(12,length(Ks));
AUC_10 = zeros(12,length(Ks));
F1_10 = zeros(12,length(Ks));

for f=1:12
    for k=1:length(Ks)
        rfMRMR_CV = fitensemble(Xtrain(:,idxMRMR(1:12)),Ytrain, 'Bag', k, 'Tree', 'Type', 'classification', 'CVPartition', cvKF10);
        [rfMRMR_pred,scores] = kfoldPredict(rfMRMR_CV);
        rfMRMR_10(f,k) = mean(Ytrain == rfMRMR_pred);
        
        [~,~,~,AUCrf_n] = perfcurve(Ytrain,scores(:,1),'Necrosis');
        [~,~,~,AUCrf_s] = perfcurve(Ytrain,scores(:,2),'Stroma');
        [~,~,~,AUCrf_t] = perfcurve(Ytrain,scores(:,3),'Tumor');
        AUC_10(f,k) = (AUCrf_n + AUCrf_s + AUCrf_t)/3;
        tpNk = sum(ismember(Ytrain,'Necrosis') & ismember(rfMRMR_pred,'Necrosis'));
        tpSk = sum(ismember(Ytrain,'Stroma') & ismember(rfMRMR_pred,'Stroma'));
        tpTk = sum(ismember(Ytrain,'Tumor') & ismember(rfMRMR_pred,'Tumor'));
        fpNk = sum(ismember(rfMRMR_pred,'Necrosis')) - tpNk;
        fpSk = sum(ismember(rfMRMR_pred,'Stroma')) - tpSk;
        fpTk = sum(ismember(rfMRMR_pred,'Tumor')) - tpTk;
        fnNk = sum(ismember(Ytrain, 'Necrosis')) - tpNk;
        fnSk = sum(ismember(Ytrain, 'Stroma')) - tpSk;
        fnTk = sum(ismember(Ytrain, 'Tumor')) - tpTk;
        f1Nk = tpNk/(tpNk + (fpNk+fnNk)/2);
        f1Sk = tpSk/(tpSk + (fpSk+fnSk)/2);
        f1Tk = tpTk/(tpTk + (fpTk+fnTk)/2);
        F1_10(f,k) = (f1Nk + f1Sk + f1Tk)/3;
    end
end
%% Heatmap Accuracy
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(xvals_numF, yvals_k, rfMRMR_10, 'Colormap', jet, 'CellLabelColor','none');
title('Random Forest - Accuracy');
h5.XLabel = 'K_R_F';
h5.YLabel = 'Num of Features';
%% Heatmap AUC
figure;
h2 = heatmap(xvals_numF, yvals_k, AUC_10, 'Colormap', jet, 'CellLabelColor','none');
title('Random Forest - AUC');
h2.XLabel = 'K_R_F';
h2.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(xvals_numF, yvals_k,F1_10, 'Colormap', jet, 'CellLabelColor','none');
title('Random Forest - F1 Score');
h.XLabel = 'K_R_F';
h.YLabel = 'Num of Features';
%%
figure;
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
subplot(3,1,1);
heatmap(xvals_numF, yvals_k, rfMRMR_10,'Colormap', jet, 'CellLabelColor','none');
title('Random Forest: Accruacy 10-Fold');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
subplot(3,1,2);
heatmap(xvals_numF, yvals_k, AUC_10,'Colormap', jet, 'CellLabelColor','none');
title('Random Forest: AUC 10-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
subplot(3,1,3);
heatmap(xvals_numF, yvals_k,F1_10,'Colormap', jet, 'CellLabelColor','none');
title('Random Forest: F1 Score 10-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% External Validtion
rfMRMR_flat_1 = reshape(rfMRMR_10,[],1);
rfMRMR_flat = sort(unique(rfMRMR_flat_1), 'descend');
fk = [];
rfMRMR_CV_preds = [];
%%%%%%%%%%%%%%%%%%%%Different feature num: Choose 3 features out of 7
for a=1:3
    inds = find(rfMRMR_10 == rfMRMR_flat(a));
    for i = 1:length(inds)
        [f,k] = ind2sub(size(rfMRMR_10),inds(i));
        fk = [fk; f, k];
        rfMRMR_CV_preds = [rfMRMR_CV_preds, rfMRMR_flat(a)];
    end
end
%%
rfMRMR_preds = [];
for i=1:length(fk)
    f = fk(i,1);
    k = fk(i,2);
    disp(Ks(k))
    rfMRMR_CV = fitensemble(Xtrain(:,idxMRMR(1:f)),Ytrain, 'Bag', Ks(k), 'Tree', 'Type', 'classification');
    pred = predict(rfMRMR_CV, Xtest(:,idxMRMR(1:f)));
    rfMRMR_preds = [rfMRMR_preds, mean(pred==Ytest)]; % accuracy
    
end

%% Plot Acuracy
figure;
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on

scatter(rfMRMR_CV_preds,rfMRMR_preds,'+','Linewidth',1);

xlabel('Internal');
ylabel('External');
title('Internal Validation vs. External Validation');
%%
save("rfMRMR_preds.mat","rfMRMR_preds");
save("rfMRMR_CV_preds.mat","rfMRMR_CV_preds");
%%
% 9, 90
master_rf = [rfMRMR_preds.', fk];
master_rf = sortrows(master_rf,1,'descend');
bf = master_rf(1,2);
bk = Ks(master_rf(1,3));
rfMRMR = fitensemble(Xtrain(:,idxMRMR(1:9)),Ytrain, 'Bag', 90, 'Tree', 'Type', 'classification');
[pred,score] = predict(rfMRMR, Xtest(:,idxMRMR(1:9)));
acc = mean(pred==Ytest) * 100 % accuracy
figure;
cm = confusionchart(cellstr(Ytest),pred);
title(['Random Forest: ', num2str(acc), '%'])
%%
figure;
subplot(2,1,1);
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on

scatter(rfMRMR_CV_preds,rfMRMR_preds,'+','Linewidth',1);

xlabel('Internal');
ylabel('External');
title('Internal Validation vs. External Validation');
master_rf = [rfMRMR_preds.', fk];
master_rf = sortrows(master_rf,1,'descend');
bf = master_rf(1,2);
bk = Ks(master_rf(1,3));
rfMRMR = fitensemble(Xtrain(:,idxMRMR(1:9)),Ytrain, 'Bag', 90, 'Tree', 'Type', 'classification');
[pred,score] = predict(rfMRMR, Xtest(:,idxMRMR(1:9)));
acc = mean(pred==Ytest) * 100 % accuracy
subplot(2,1,2);
confusionchart(cellstr(Ytest),pred);
title(['Random Forest: ', num2str(acc), '%'])

%% AUC
[X_rf_n,Y_rf_n,T,AUCrf_n] = perfcurve(Ytest,score(:,1),'Necrosis');
[X_rf_s,Y_rf_s,T,AUCrf_s] = perfcurve(Ytest,score(:,2),'Stroma');
[X_rf_t,Y_rf_t,T,AUCrf_t] = perfcurve(Ytest,score(:,3),'Tumor');









%% Load Dataset
labels = load("labels.mat").labels;
total = load("totalFeatures.mat").total;
featureNames = load("totalFeaturesNames.mat").featureNames;
%% Split
% Split data into Train & Test
cv = cvpartition(size(total, 1), 'HoldOut', 0.3);
idx = cv.test;
Xtrain = total(~idx,:);
Ytrain = labels(~idx,:);
Xtest = total(idx,:);
Ytest = labels(idx,:);
save('Xtrain.mat', 'Xtrain');
save('Ytrain.mat', 'Ytrain');
save('Xtest.mat', 'Xtest');
save('Ytest.mat', 'Ytest');
%% Split - K-Fold
% Split data into K-Fold(10)
cvKF5 = cvpartition(size(Xtrain, 1), 'KFold', 5);
cvKF10 = cvpartition(size(Xtrain, 1), 'KFold', 10);
cvKF20 = cvpartition(size(Xtrain, 1), 'KFold', 20);
save('cvKF5.mat','cvKF5');
save('cvKF10.mat','cvKF10');
save('cvKF20.mat','cvKF20');
% cvKF5 = load('cvKF5.mat','cvKF5').cvKF5;
% cvKF10 = load('cvKF10.mat','cvKF10').cvKF10;
% cvKF20 = load('cvKF20.mat','cvKF20').cvKF20;
load ionosphere
%% MRMR :)
[idxMRMR, scores] = fscmrmr(total, labels);
MRMR = [idxMRMR; scores];
save('MRMR.mat', 'MRMR');
%% MRMR - Score Analysis :)
figure; plot([1:length(scores)], scores(idxMRMR))
title("MRMR Scores")
xlabel("Number of Features")
ylabel("Score")
%% KNN(MRMR): Find Optimal K :)
% Accuracy
knnMRMR_preds = [];
Ks = [1:2:51];
for k=Ks
    knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:12)),Ytrain,'NumNeighbors',k,'Standardize',1);
    knnMRMR_pred = predict(knnMRMR, Xtest(:, idxMRMR(1:12)));
    knnMRMR_preds = [knnMRMR_preds, mean(knnMRMR_pred == Ytest)];
end
%% KNN(MRMR): Visualiztion - Accuracy :)
figure; plot(Ks, knnMRMR_preds);
title("KNN w/ MRMR")
xlabel("k")
ylabel("Accuracy")
disp(max(knnMRMR_preds));
%% KNN(MRMR): Find Optimal K :)
% Loss: Using K-Fold
knnMRMR_losses5 = [];
knnMRMR_losses10 = [];
knnMRMR_losses20 = [];
Ks =[1:2:51];
for k=Ks
    knnMRMR_CV = fitcknn(total(:,1:12),labels,'NumNeighbors',k,'Standardize',1,'CVPartition',cvKF5);
    knnMRMR_losses5 = [knnMRMR_losses5, kfoldLoss(knnMRMR_CV)];
end
for k=Ks
    knnMRMR_CV = fitcknn(total(:,1:12),labels,'NumNeighbors',k,'Standardize',1,'CVPartition',cvKF10);
    knnMRMR_losses10 = [knnMRMR_losses10, kfoldLoss(knnMRMR_CV)];
end
for k=Ks
    knnMRMR_CV = fitcknn(total(:,1:12),labels,'NumNeighbors',k,'Standardize',1,'CVPartition',cvKF20);
    knnMRMR_losses20 = [knnMRMR_losses20, kfoldLoss(knnMRMR_CV)];
end
%% KNN(MRMR): Visualization - Loss :)
figure; plot(Ks, knnMRMR_losses5);
title("KNN MRMR(K-Fold = 5)")
xlabel("K")
ylabel("Loss")
figure; plot(Ks, knnMRMR_losses10);
title("KNN MRMR(K-Fold = 10)")
xlabel("K")
ylabel("Loss")
figure; plot(Ks, knnMRMR_losses20);
title("KNN MRMR(K-Fold = 20)")
xlabel("K")
ylabel("Loss")
%% KNN(MRMR): Result Analysis :)
% Accuracy 0.9859
knnMRMR_best = fitcknn(total(:,1:12), labels,'NumNeighbors',7,'Standardize',1, 'CVPartition',cvKF10);
knnMRMR_pred_best = kfoldPredict(knnMRMR_best);
figure;
cm10 = confusionchart(cellstr(labels), knnMRMR_pred_best);
title('Confusion Matrix: KNN MRMR(K-Fold = 10)')
knnMRMR_best20 = fitcknn(total(:,1:12), labels,'NumNeighbors',7,'Standardize',1, 'CVPartition',cvKF20);
knnMRMR_pred_best20 = kfoldPredict(knnMRMR_best);
figure;
cm20 = confusionchart(cellstr(labels), knnMRMR_pred_best);
title('Confusion Matrix: KNN MRMR(K-Fold = 20)')
disp(mean(knnMRMR_pred_best20 == labels))
knnMRMR_best5 = fitcknn(total(:,1:12), labels,'NumNeighbors',7,'Standardize',1, 'CVPartition',cvKF5);
knnMRMR_pred_best20 = kfoldPredict(knnMRMR_best);
figure;
cm5 = confusionchart(cellstr(labels), knnMRMR_pred_best);
title('Confusion Matrix: KNN MRMR(K-Fold = 5)')
disp(mean(knnMRMR_pred_best20 == labels))
%% SVM(MRMR) - Train
% No need to do the KFold I think? Only validation that makes sense for
% SVM is holdout which is spliting data into Train & Test.
% Accuracy = 0.6244
svmMRMR = fitcecoc(total(:,1:12), labels, 'CVPartition',cvKF10);
svmMRMR_pred = kfoldPredict(svmMRMR);
figure;
%%
cm = confusionchart(cellstr(labels), knnMRMR_pred_best);
title('Confusion Matrix: SVM MRMR(K-Fold = 10)')
disp(mean(labels == svmMRMR_pred));
%% Random Forest
% 0.9953
rfMRMR = fitensemble(total(:,idxMRMR(1:12)),labels, 'Bag', 100, 'Tree', 'Type', 'classification', 'CVPartition', cvKF10);
rfMRMR_pred = kfoldPredict(rfMRMR);
disp(mean(rfMRMR_pred == labels));
%% PCA
[coeff, score, latent, ~, explained, mu] = pca(Xtrain);
%% PCA - Compute Input Features
idx = find(cumsum(explained)>95,1);
scoreTrain95 = score(:,1:idx);
%% SVM(PCA) - Train
svmPCA = fitcecoc(scoreTrain95,Ytrain);
%% SVM(PCA) - Predict
% accuracy:0.3153  |  variance% = 95
scoreTest95 = (Xtest-mu)*coeff(:,1:idx);
svmPCA_pred = predict(svmPCA,scoreTest95);
disp(mean(svmPCA_pred == Ytest));
%% KNN(PCA) - Train
knnPCA = fitcknn(scoreTrain95,Ytrain,'NumNeighbors',7,'Standardize',1);
%% KNN(PCA) - Predict
% accuracy:0.3421  |  variance% = 95  NumNeighbors:7  Standardize:on
knnPCA_pred = predict(knnPCA, scoreTest95);
disp(mean(knnPCA_pred == Ytest));
%%
%%Random Forest
rfPCA = fitensemble(scoreTrain95,Ytrain, 'Bag', 100, 'Tree', 'Type', 'classification');
rfPCA_pred = predict(rfPCA,scoreTest95);
disp(mean(rfMRMR_pred ==Ytest));
%%
t = templateSVM('KernelFunction','polynomial','PolynomialOrder',3);
Mdl = fitcecoc(Xtrain(:,idxMRMR(1:12)),Ytrain,'Learners',t);
pred = predict(Mdl, Xtest(1:12));
disp(mean(pred == Ytest));
%%
Mdl = fitcnb(total(:,idxMRMR(1:5)),labels, 'CVPartition', cvKF10);
pred = kfoldPredict(Mdl);
disp(mean(pred == labels));
confusionchart(cellstr(labels), pred);






clear;
clc;
format long
%% Load Dataset
Xtrain = load('Xtrain.mat').Xtrain;
Ytrain = load('Ytrain.mat').Ytrain;
Xtest = load('Xtest.mat').Xtest;
Ytest = load('Ytest.mat').Ytest;
cvKF5 = load('cvKF5.mat','cvKF5').cvKF5;
cvKF10 = load('cvKF10.mat','cvKF10').cvKF10;
cvKF20 = load('cvKF20.mat','cvKF20').cvKF20;
%% Load MRMR
MRMR = load('MRMR.mat').MRMR;
idxMRMR = MRMR(1,:);
scores = MRMR(2,:);
%% K vs Num. Features (K-Fold = 5)
Ks =[1:2:51];
knnMRMR_5 = zeros(12,length(Ks));
AUC_5 = zeros(12,length(Ks));
F1_5 = zeros(12,length(Ks));

for f=1:12
    for k=1:length(Ks)
        knnMRMR_CV = fitcknn(Xtrain(:,idxMRMR(1:f)),Ytrain,'NumNeighbors',k,'Standardize',1,'CVPartition',cvKF5);
        [knnMRMR_pred, scores] = kfoldPredict(knnMRMR_CV);
        knnMRMR_5(f,k) = mean(Ytrain == knnMRMR_pred);
        [~,~,~,AUCknn_n] = perfcurve(Ytrain,scores(:,1),'Necrosis');
        [~,~,~,AUCknn_s] = perfcurve(Ytrain,scores(:,2),'Stroma');
        [~,~,~,AUCknn_t] = perfcurve(Ytrain,scores(:,3),'Tumor');
        AUC_5(f,k) = (AUCknn_n + AUCknn_s + AUCknn_t)/3;
        tpNk = sum(ismember(Ytrain,'Necrosis') & ismember(knnMRMR_pred,'Necrosis'));
        tpSk = sum(ismember(Ytrain,'Stroma') & ismember(knnMRMR_pred,'Stroma'));
        tpTk = sum(ismember(Ytrain,'Tumor') & ismember(knnMRMR_pred,'Tumor'));
        fpNk = sum(ismember(knnMRMR_pred,'Necrosis')) - tpNk;
        fpSk = sum(ismember(knnMRMR_pred,'Stroma')) - tpSk;
        fpTk = sum(ismember(knnMRMR_pred,'Tumor')) - tpTk;
        fnNk = sum(ismember(Ytrain, 'Necrosis')) - tpNk;
        fnSk = sum(ismember(Ytrain, 'Stroma')) - tpSk;
        fnTk = sum(ismember(Ytrain, 'Tumor')) - tpTk;
        f1Nk = tpNk/(tpNk + (fpNk+fnNk)/2);
        f1Sk = tpSk/(tpSk + (fpSk+fnSk)/2);
        f1Tk = tpTk/(tpTk + (fpTk+fnTk)/2);
        F1_5(f,k) = (f1Nk + f1Sk + f1Tk)/3;

    end
end
%% Heatmap Accuracy
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(xvals_numF, yvals_k, knnMRMR_5,'Colormap', jet, 'CellLabelColor','none');
title('KNN: Accruacy 5-Fold');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';

%% Heatmap AUC
figure;
h = heatmap(xvals_numF, yvals_k, AUC_5,'Colormap', jet, 'CellLabelColor','none');
title('KNN: AUC 5-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(xvals_numF, yvals_k,F1_5,'Colormap', jet, 'CellLabelColor','none');
title('KNN: F1 Score 5-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%%
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
subplot(3,1,1);
heatmap(xvals_numF, yvals_k, knnMRMR_5,'Colormap', jet, 'CellLabelColor','none');
title('KNN: Accruacy 5-Fold');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
subplot(3,1,2);
heatmap(xvals_numF, yvals_k, AUC_5,'Colormap', jet, 'CellLabelColor','none');
title('KNN: AUC 5-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
subplot(3,1,3);
heatmap(xvals_numF, yvals_k,F1_5,'Colormap', jet, 'CellLabelColor','none');
title('KNN: F1 Score 5-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% K vs Num. Features (K-Fold = 10)
Ks =[1:2:51];
knnMRMR_10 = zeros(12,length(Ks));
AUC_10 = zeros(12,length(Ks));
F1_10 = zeros(12,length(Ks));

for f=1:12
    for k=1:length(Ks)
        knnMRMR_CV = fitcknn(Xtrain(:,idxMRMR(1:f)),Ytrain,'NumNeighbors',k,'Standardize',1,'CVPartition',cvKF10);
        [knnMRMR_pred, scores] = kfoldPredict(knnMRMR_CV);
        knnMRMR_10(f,k) = mean(Ytrain == knnMRMR_pred);
        [~,~,~,AUCknn_n] = perfcurve(Ytrain,scores(:,1),'Necrosis');
        [~,~,~,AUCknn_s] = perfcurve(Ytrain,scores(:,2),'Stroma');
        [~,~,~,AUCknn_t] = perfcurve(Ytrain,scores(:,3),'Tumor');
        AUC_10(f,k) = (AUCknn_n + AUCknn_s + AUCknn_t)/3;
        tpNk = sum(ismember(Ytrain,'Necrosis') & ismember(knnMRMR_pred,'Necrosis'));
        tpSk = sum(ismember(Ytrain,'Stroma') & ismember(knnMRMR_pred,'Stroma'));
        tpTk = sum(ismember(Ytrain,'Tumor') & ismember(knnMRMR_pred,'Tumor'));
        fpNk = sum(ismember(knnMRMR_pred,'Necrosis')) - tpNk;
        fpSk = sum(ismember(knnMRMR_pred,'Stroma')) - tpSk;
        fpTk = sum(ismember(knnMRMR_pred,'Tumor')) - tpTk;
        fnNk = sum(ismember(Ytrain, 'Necrosis')) - tpNk;
        fnSk = sum(ismember(Ytrain, 'Stroma')) - tpSk;
        fnTk = sum(ismember(Ytrain, 'Tumor')) - tpTk;
        f1Nk = tpNk/(tpNk + (fpNk+fnNk)/2);
        f1Sk = tpSk/(tpSk + (fpSk+fnSk)/2);
        f1Tk = tpTk/(tpTk + (fpTk+fnTk)/2);
        F1_10(f,k) = (f1Nk + f1Sk + f1Tk)/3;

    end
end
%% Heatmap Accuracy
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(xvals_numF, yvals_k, knnMRMR_10,'Colormap', jet, 'CellLabelColor','none');
title('Acuuracy 10-Fold');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
%% Heatmap AUC
figure;
h2 = heatmap(xvals_numF, yvals_k, AUC_10,'Colormap', jet, 'CellLabelColor','none');
title('AUC 10-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(xvals_numF, yvals_k,F1_10,'Colormap', jet, 'CellLabelColor','none');
title('F1 Score 10-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%%
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
subplot(3,1,1);
heatmap(xvals_numF, yvals_k, knnMRMR_10,'Colormap', jet, 'CellLabelColor','none');
title('KNN: Accruacy 10-Fold');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
subplot(3,1,2);
heatmap(xvals_numF, yvals_k, AUC_10,'Colormap', jet, 'CellLabelColor','none');
title('KNN: AUC 10-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
subplot(3,1,3);
heatmap(xvals_numF, yvals_k,F1_10,'Colormap', jet, 'CellLabelColor','none');
title('KNN: F1 Score 10-Fold');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% External Validtion
knnMRMR_flat_1 = reshape(knnMRMR_10,[],1);
knnMRMR_flat = sort(unique(knnMRMR_flat_1), 'descend');
fk = [];
knnMRMR_CV_preds = [];
for a=1:10
    inds = find(knnMRMR_10 == knnMRMR_flat(a));
    for i = 1:length(inds)
        [f,k] = ind2sub(size(knnMRMR_10),inds(i));
        fk = [fk; f, k];
        knnMRMR_CV_preds = [knnMRMR_CV_preds, knnMRMR_flat(a)];
    end
end
%%
knnMRMR_preds = [];
for i=1:length(fk)
    f = fk(i,1);
    k = fk(i,2);
    knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:f)),Ytrain,'NumNeighbors',Ks(k),'Standardize',1);
    pred = predict(knnMRMR, Xtest(:,idxMRMR(1:f)));
    knnMRMR_preds = [knnMRMR_preds, mean(pred==Ytest)]; % accuracy
    
end

%% Plot Acuracy
figure;
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on
scatter(knnMRMR_CV_preds,knnMRMR_preds,'+','Linewidth',1);
title('Internal Validation vs. External Validation');
xlabel('Internal');
ylabel('External');


%% CM
% # features: 12, k: 3
master = [knnMRMR_preds.', fk];
master = sortrows(master,1,'descend');
bf = master(1,2);
bk = Ks(master(1,3));
knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:bf)),Ytrain,'NumNeighbors',3,'Standardize',1);
[pred,score,~] = predict(knnMRMR, Xtest(:,idxMRMR(1:bf)));
acc = mean(pred==Ytest) * 100 % accuracy
figure;
cm = confusionchart(cellstr(Ytest),pred);
title(['KNN: ', num2str(acc), '%'])
%%
subplot(2,1,1);
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on
scatter(knnMRMR_CV_preds,knnMRMR_preds,'+','Linewidth',1);
title('Internal Validation vs. External Validation');
xlabel('Internal');
ylabel('External');
subplot(2,1,2);
master = [knnMRMR_preds.', fk];
master = sortrows(master,1,'descend');
bf = master(1,2);
bk = Ks(master(1,3));
knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:bf)),Ytrain,'NumNeighbors',3,'Standardize',1);
[pred,score,~] = predict(knnMRMR, Xtest(:,idxMRMR(1:bf)));
acc = mean(pred==Ytest) * 100 % accuracy
confusionchart(cellstr(Ytest),pred);
title(['KNN: ', num2str(acc), '%'])
%% AUC
[X_knn_n,Y_knn_n,T,AUCknn_n] = perfcurve(Ytest,score(:,1),'Necrosis');
[X_knn_s,Y_knn_s,T,AUCknn_s] = perfcurve(Ytest,score(:,2),'Stroma');
[X_knn_t,Y_knn_t,T,AUCknn_t] = perfcurve(Ytest,score(:,3),'Tumor');
