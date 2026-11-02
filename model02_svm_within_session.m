
%% MODEL 02 - RBF SUPPORT VECTOR MACHINE
% Features: F1-F5
% Within-session 70/30 classification
% Reuses exact Model 01 train/test partitions
% Training-only normalization

clearvars; clc;
rng(42);

root = 'D:\Project';
outDir = fullfile(root,'Results');

assert(exist('fitcsvm','file')==2, ...
    'Statistics and Machine Learning Toolbox required.');

%% Load dataset and saved partition

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'));

P = readtable(fullfile(outDir, ...
    'within_session_split_rng42.csv'), ...
    'Delimiter', ',', ...
    'VariableNamingRule', 'preserve');

% Verify the saved partition was loaded correctly
assert(height(P) == 609, ...
    'Expected 609 partition records.');

assert(all(ismember( ...
    {'RecordID','Capture','Partition'}, ...
    P.Properties.VariableNames)), ...
    'Partition column names are incorrect.');

fprintf('Partition loaded: %d rows\n', height(P));

featureNames = {
    'F1_transient_energy'
    'F2_mean_amplitude'
    'F3_amplitude_variance'
    'F4_residual_phase_variance'
    'F5_tone_offset_hz'
};

X = T{:,featureNames};

y = string(T.device_id);
session = string(T.session_id);
status = string(T.quality_status);

classes = ["D2";"D3"];

%% Match saved partitions by RecordID

recordIDs = string(T.record_id);
savedIDs = string(P.RecordID);

assert(numel(unique(savedIDs))==numel(savedIDs), ...
    'Duplicate RecordIDs in partition file.');

[found,loc] = ismember(recordIDs,savedIDs);

assert(all(found) && height(P)==height(T), ...
    'Saved partition does not match the dataset.');

splitRole = string(P.Partition(loc));

eligible = status=="ACCEPT" & ...
    ismember(y,classes) & all(isfinite(X),2);

assert(all(ismember(splitRole(eligible), ...
    ["TRAIN","TEST"])), ...
    'Eligible rows are missing partition assignments.');

%% Evaluation

fprintf('\n===== MODEL 02: RBF-SVM (F1-F5) =====\n');

summaryResults = table();
allPredictions = table();

for ss = ["S01","S02"]

    train = eligible & session==ss & ...
        splitRole=="TRAIN";

    test = eligible & session==ss & ...
        splitRole=="TEST";

    assert(sum(train)>0 && sum(test)>0, ...
        'Empty training or testing set.');

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    assert(all(ismember(classes,unique(ytr))), ...
        'Training must contain both devices.');

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

    %% Predict

    ypred = string(predict(mdl,Zte));

    %% Confusion matrix

    C = zeros(2,2);

    for i = 1:2
        for j = 1:2
            C(i,j) = sum(yte==classes(i) & ...
                         ypred==classes(j));
        end
    end

    %% Metrics

    accuracy = 100*sum(diag(C))/sum(C(:));

    precision = zeros(2,1);
    recall = zeros(2,1);
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

        if precision(c)+recall(c)>0
            F1Score(c) = ...
                2*precision(c)*recall(c)/ ...
                (precision(c)+recall(c));
        end
    end

    balanced = 100*mean(recall);

    %% Display results

    fprintf('\n===== SESSION %s =====\n',ss);

    fprintf('Training: %d events\n',sum(train));
    fprintf('Testing: %d events\n',sum(test));

    disp(array2table(C, ...
        'VariableNames',{'Pred_D2','Pred_D3'}, ...
        'RowNames',{'Actual_D2','Actual_D3'}));

    fprintf('Accuracy: %.2f%%\n',accuracy);
    fprintf('Balanced accuracy: %.2f%%\n',balanced);

    disp(table(classes,precision,recall,F1Score, ...
        'VariableNames', ...
        {'Device','Precision','Recall','F1Score'}));

    fprintf('Kernel scale: %.6g\n', ...
        mdl.KernelParameters.Scale);

    fprintf('Support vectors: %d\n', ...
        size(mdl.SupportVectors,1));

    %% Save trained model and normalization

    modelFile = fullfile(outDir, ...
        sprintf('model02_svm_%s.mat',char(ss)));

    save(modelFile,'mdl','mu','sd', ...
        'featureNames','classes');

    %% Record summary

    result = table(ss,sum(train),sum(test), ...
        accuracy,balanced,recall(1),recall(2), ...
        'VariableNames', ...
        {'Session','TrainCount','TestCount', ...
         'AccuracyPercent','BalancedAccuracyPercent', ...
         'D2Recall','D3Recall'});

    summaryResults = [summaryResults;result];

    %% Record individual test predictions

    testIDs = recordIDs(test);
    testSession = repmat(ss,numel(yte),1);

    predTable = table(testIDs,testSession,yte,ypred, ...
        'VariableNames', ...
        {'RecordID','Session','Actual','Predicted'});

    allPredictions = [allPredictions;predTable];

end

%% Export derived results

writetable(summaryResults,fullfile(outDir, ...
    'model02_svm_within_session_5f.csv'));

writetable(allPredictions,fullfile(outDir, ...
    'model02_svm_predictions_5f.csv'));

fprintf('\n===== MODEL 02 SUMMARY =====\n');
disp(summaryResults);

fprintf('\nResults and models saved in Results folder.\n');
fprintf('===== MODEL 02 COMPLETE =====\n');
