
%% BASELINE 01 - Session-separated RF classification
% Training: S01 | Testing: S02
% No source files modified. No toolboxes required.

clearvars; clc;

root = 'D:\Project';

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'));

features = {
    'F2_mean_amplitude'
    'F3_amplitude_variance'
    'F4_residual_phase_variance'
    'F5_tone_offset_hz'
};

X = T{:,features};

device = string(T.device_id);
session = string(T.session_id);
status = string(T.quality_status);

train = session=="S01" & status=="ACCEPT";
test  = session=="S02" & status=="ACCEPT";

assert(all(isfinite(X(:))), ...
    'Non-finite feature values detected.');

assert(any(train) && any(test), ...
    'Training or testing partition is empty.');

classes = ["D2";"D3"];

fprintf('\n===== SESSION HOLDOUT SETUP =====\n');
fprintf('Training S01: %d rows\n',sum(train));
fprintf('Testing S02: %d rows\n',sum(test));

for c = 1:2
    fprintf('%s: Train %d | Test %d\n',classes(c), ...
        sum(train & device==classes(c)), ...
        sum(test & device==classes(c)));

    assert(any(train & device==classes(c)), ...
        'Class absent from training set.');
end

%% Training-only standardization

Xtrain = X(train,:);
Xtest = X(test,:);

mu = mean(Xtrain,1);
sd = std(Xtrain,0,1);

sd(sd==0) = 1;

Ztrain = (Xtrain-mu)./sd;
Ztest  = (Xtest-mu)./sd;

ytrain = device(train);
ytest = device(test);

%% Nearest-centroid model

centroids = zeros(2,size(Xtrain,2));

for c = 1:2
    centroids(c,:) = mean( ...
        Ztrain(ytrain==classes(c),:),1);
end

distances = zeros(size(Ztest,1),2);

for c = 1:2
    distances(:,c) = sum( ...
        (Ztest-centroids(c,:)).^2,2);
end

[~,predictionIdx] = min(distances,[],2);
ypred = classes(predictionIdx);

%% Confusion matrix

C = zeros(2,2);

for i = 1:2
    for j = 1:2
        C(i,j) = sum(ytest==classes(i) & ...
                     ypred==classes(j));
    end
end

fprintf('\n===== CONFUSION MATRIX =====\n');
fprintf('Rows = Actual | Columns = Predicted\n');

disp(array2table(C, ...
    'VariableNames',{'Pred_D2','Pred_D3'}, ...
    'RowNames',{'Actual_D2','Actual_D3'}));

%% Classification metrics

accuracy = sum(diag(C))/sum(C(:));

precision = nan(2,1);
recall = nan(2,1);
f1 = nan(2,1);

for c = 1:2

    TP = C(c,c);
    predicted = sum(C(:,c));
    actual = sum(C(c,:));

    if predicted>0
        precision(c) = TP/predicted;
    end

    if actual>0
        recall(c) = TP/actual;
    end

    if isfinite(precision(c)) && ...
       precision(c)+recall(c)>0
        f1(c) = 2*precision(c)*recall(c)/ ...
            (precision(c)+recall(c));
    elseif actual>0 && predicted==0
        f1(c) = 0;
    end
end

fprintf('\n===== METRICS =====\n');
fprintf('Accuracy: %.2f%%\n',100*accuracy);
fprintf('Balanced accuracy: %.2f%%\n', ...
    100*mean(recall));

disp(table(classes,precision,recall,f1, ...
    'VariableNames', ...
    {'Device','Precision','Recall','F1Score'}));

fprintf('\n===== MAJORITY BASELINE =====\n');

fprintf('Always predict D2: %.2f%% accuracy\n', ...
    100*mean(ytest=="D2"));

fprintf('Always predict D3: %.2f%% accuracy\n', ...
    100*mean(ytest=="D3"));

fprintf('\n===== BASELINE COMPLETE =====\n');
