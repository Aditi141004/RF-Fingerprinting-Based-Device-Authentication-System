
%% MODEL 04 - KNN LEAVE-ONE-CAPTURE-OUT
% F1-F5 | k=5 | Euclidean distance
% Training-only normalization
% Six held-out capture folds

clearvars; clc;

root = fileparts(mfilename('fullpath'));
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% Load dataset

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'), ...
    'VariableNamingRule','preserve');

featureNames = {
    'F1_transient_energy'
    'F2_mean_amplitude'
    'F3_amplitude_variance'
    'F4_residual_phase_variance'
    'F5_tone_offset_hz'
};

X = T{:,featureNames};
y = string(T.device_id);

keys = string(T.device_id) + "/" + ...
       string(T.session_id) + "/" + ...
       string(T.capture_id);

classes = ["D2";"D3"];

eligible = string(T.quality_status)=="ACCEPT" & ...
    ismember(y,classes) & all(isfinite(X),2);

folds = unique(keys(eligible),'stable');

assert(numel(folds)==6, ...
    'Expected six accepted captures.');

assert(sum(eligible)==609, ...
    'Expected 609 accepted events.');

pred = strings(height(T),1);

nTest = zeros(numel(folds),1);
nCorrect = zeros(numel(folds),1);
foldAccuracy = zeros(numel(folds),1);

fprintf('\n===== MODEL 04: KNN LOCO =====\n');

%% Leave one entire capture out per fold

for f = 1:numel(folds)

    test = eligible & keys==folds(f);
    train = eligible & keys~=folds(f);

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    assert(all(ismember(classes,unique(ytr))), ...
        'Both devices must appear in training.');

    %% Training-only standardization

    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0) = 1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    %% Train k-NN

    mdl = fitcknn(Ztr,categorical(ytr), ...
        'NumNeighbors',5, ...
        'Distance','euclidean', ...
        'DistanceWeight','equal', ...
        'Standardize',false);

    %% Predict held-out capture

    ypred = string(predict(mdl,Zte));

    pred(test) = ypred;

    nTest(f) = sum(test);
    nCorrect(f) = sum(ypred==yte);

    foldAccuracy(f) = ...
        100*nCorrect(f)/nTest(f);

    fprintf('%s: %.2f%% (%d/%d)\n', ...
        char(folds(f)),foldAccuracy(f), ...
        nCorrect(f),nTest(f));
end

%% Aggregate confusion matrix

C = zeros(2,2);

for i = 1:2
    for j = 1:2
        C(i,j) = sum(eligible & ...
            y==classes(i) & pred==classes(j));
    end
end

fprintf('\n===== AGGREGATE CONFUSION MATRIX =====\n');

disp(array2table(C, ...
    'VariableNames',{'Pred_D2','Pred_D3'}, ...
    'RowNames',{'Actual_D2','Actual_D3'}));

%% Classification metrics

accuracy = 100*sum(diag(C))/sum(C(:));

precision = nan(2,1);
recall = nan(2,1);
F1Score = zeros(2,1);

for c = 1:2

    TP = C(c,c);
    actual = sum(C(c,:));
    predicted = sum(C(:,c));

    if actual>0
        recall(c) = TP/actual;
    end

    if predicted>0
        precision(c) = TP/predicted;
    end

    if isfinite(precision(c)) && ...
       precision(c)+recall(c)>0

        F1Score(c) = ...
            2*precision(c)*recall(c)/ ...
            (precision(c)+recall(c));
    end
end

balanced = 100*mean(recall);

fprintf('\n===== OVERALL METRICS =====\n');

fprintf('Accuracy: %.2f%%\n',accuracy);
fprintf('Balanced accuracy: %.2f%%\n',balanced);

disp(table(classes,precision,recall,F1Score, ...
    'VariableNames', ...
    {'Device','Precision','Recall','F1Score'}));

%% Save fold results and predictions

foldTable = table(folds,nTest,nCorrect,foldAccuracy, ...
    'VariableNames', ...
    {'Capture','TestCount','Correct','AccuracyPercent'});

writetable(foldTable,fullfile(outDir, ...
    'model04_knn_loco_5f.csv'));

predictionTable = table( ...
    string(T.record_id(eligible)), ...
    keys(eligible), ...
    y(eligible), ...
    pred(eligible), ...
    'VariableNames', ...
    {'RecordID','Capture','Actual','Predicted'});

writetable(predictionTable,fullfile(outDir, ...
    'model04_knn_loco_predictions_5f.csv'));

fprintf('\nResults saved in Results folder.\n');
fprintf('===== MODEL 04 LOCO COMPLETE =====\n');
