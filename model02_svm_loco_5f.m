
%% MODEL 02 - RBF-SVM LEAVE-ONE-CAPTURE-OUT
% All five features: F1-F5
% Six capture-separated folds
% Training-only Z-score normalization

clearvars; clc;
rng(42);

root = 'D:\Project';
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'), ...
    'VariableNamingRule','preserve');

%% Dataset preparation

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

pred = strings(height(T),1);

foldAccuracy = zeros(numel(folds),1);
nTest = zeros(numel(folds),1);
nCorrect = zeros(numel(folds),1);

fprintf('\n===== MODEL 02: RBF-SVM LOCO =====\n');

%% Six capture-separated folds

for f = 1:numel(folds)

    test = eligible & keys==folds(f);
    train = eligible & keys~=folds(f);

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    assert(all(ismember(classes,unique(ytr))), ...
        'Both training classes are required.');

    %% Training-only standardization

    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0) = 1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    %% Train SVM

    mdl = fitcsvm(Ztr,categorical(ytr), ...
        'KernelFunction','rbf', ...
        'KernelScale','auto', ...
        'BoxConstraint',1, ...
        'Standardize',false);

    %% Predict unseen capture

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

%% Overall metrics

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

fprintf('\n===== OVERALL METRICS =====\n');

fprintf('Accuracy: %.2f%%\n',accuracy);
fprintf('Balanced accuracy: %.2f%%\n', ...
    100*mean(recall));

disp(table(classes,precision,recall,F1Score, ...
    'VariableNames', ...
    {'Device','Precision','Recall','F1Score'}));

%% Save results

foldTable = table(folds,nTest,nCorrect,foldAccuracy, ...
    'VariableNames', ...
    {'Capture','TestCount','Correct','AccuracyPercent'});

writetable(foldTable,fullfile(outDir, ...
    'model02_svm_loco_5f.csv'));

predictionTable = table( ...
    string(T.record_id(eligible)), ...
    keys(eligible), ...
    y(eligible), ...
    pred(eligible), ...
    'VariableNames', ...
    {'RecordID','Capture','Actual','Predicted'});

writetable(predictionTable,fullfile(outDir, ...
    'model02_svm_loco_predictions_5f.csv'));

fprintf('\nResults saved in Results folder.\n');
fprintf('===== MODEL 02 LOCO COMPLETE =====\n');
