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