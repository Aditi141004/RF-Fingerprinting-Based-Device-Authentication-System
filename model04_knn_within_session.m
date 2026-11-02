
%% MODEL 04 - K-NEAREST NEIGHBORS (KNN)
% Five RF features: F1-F5
% k = 5, Euclidean distance
% Same saved train/test partitions as Models 01-03
% Training-only Z-score standardization

clearvars; clc;

root = fileparts(mfilename('fullpath'));
outDir = fullfile(root,'Results');

%% Load dataset and existing split

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'), ...
    'VariableNamingRule','preserve');

P = readtable(fullfile(outDir, ...
    'within_session_split_rng42.csv'), ...
    'Delimiter',',', ...
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
session = string(T.session_id);
status = string(T.quality_status);
classes = ["D2";"D3"];

%% Verify saved train/test partitions

recordIDs = string(T.record_id);
savedIDs = string(P.RecordID);

assert(height(P)==height(T), ...
    'Partition size mismatch.');

assert(numel(unique(savedIDs))==height(P), ...
    'Duplicate record IDs in saved split.');

[found,loc] = ismember(recordIDs,savedIDs);

assert(all(found), ...
    'Saved partition does not match dataset.');

splitRole = string(P.Partition(loc));

captureKeys = y + "/" + session + "/" + ...
    string(T.capture_id);

assert(all(string(P.Capture(loc))==captureKeys), ...
    'Capture identifier mismatch.');

eligible = status=="ACCEPT" & ...
    ismember(y,classes) & all(isfinite(X),2);

assert(all(ismember(splitRole(eligible), ...
    ["TRAIN","TEST"])), ...
    'Invalid partition assignment.');

%% Evaluate each session

fprintf('\n===== MODEL 04: KNN (F1-F5) =====\n');

summaryResults = table();
allPredictions = table();

for ss = ["S01","S02"]

    train = eligible & session==ss & ...
        splitRole=="TRAIN";

    test = eligible & session==ss & ...
        splitRole=="TEST";

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    assert(all(ismember(classes,unique(ytr))), ...
        'Both classes must appear in training.');

    %% Training-only normalization

    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0)=1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    %% Train k-NN classifier

    mdl = fitcknn(Ztr,categorical(ytr), ...
        'NumNeighbors',5, ...
        'Distance','euclidean', ...
        'DistanceWeight','equal', ...
        'Standardize',false);

    %% Test predictions

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

    %% Save model and preprocessing

    save(fullfile(outDir, ...
        sprintf('model04_knn_%s.mat',char(ss))), ...
        'mdl','mu','sd','featureNames','classes');

    %% Save session summary

    result = table(ss,sum(train),sum(test), ...
        accuracy,balanced,recall(1),recall(2), ...
        'VariableNames', ...
        {'Session','TrainCount','TestCount', ...
         'AccuracyPercent','BalancedAccuracyPercent', ...
         'D2Recall','D3Recall'});

    summaryResults = [summaryResults;result];

    predTable = table( ...
        recordIDs(test), ...
        repmat(ss,sum(test),1), ...
        yte,ypred, ...
        'VariableNames', ...
        {'RecordID','Session','Actual','Predicted'});

    allPredictions = [allPredictions;predTable];
end

%% Export results

writetable(summaryResults,fullfile(outDir, ...
    'model04_knn_within_session_5f.csv'));

writetable(allPredictions,fullfile(outDir, ...
    'model04_knn_predictions_5f.csv'));

fprintf('\n===== MODEL 04 SUMMARY =====\n');
disp(summaryResults);

fprintf('\n===== MODEL 04 COMPLETE =====\n');
