
%% MODEL 01 - NCC CROSS-SESSION (ALL FIVE FEATURES)
% Final five-feature evaluation
% Training-only Z-score standardization
% Train S01 -> Test S02
% Train S02 -> Test S01

clearvars; clc;

root = 'D:\Project';
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'));

%% Final feature selection

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

eligible = status=="ACCEPT" & ...
    ismember(y,classes) & all(isfinite(X),2);

trainSessions = ["S01";"S02"];
testSessions = ["S02";"S01"];

summaryResults = table();

fprintf('\n===== MODEL 01: NCC (F1-F5) =====\n');
fprintf('Evaluation: Cross-session\n');

for fold = 1:2

    trSession = trainSessions(fold);
    teSession = testSessions(fold);

    train = eligible & session==trSession;
    test = eligible & session==teSession;

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    assert(~isempty(ytr) && ~isempty(yte), ...
        'Training or test set is empty.');

    %% Training-only standardization

    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0) = 1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    %% Train Nearest-Centroid Classifier

    centroids = zeros(2,size(X,2));

    for c = 1:2
        assert(any(ytr==classes(c)), ...
            'Training class missing.');

        centroids(c,:) = ...
            mean(Ztr(ytr==classes(c),:),1);
    end

    %% Predict

    distances = zeros(size(Zte,1),2);

    for c = 1:2
        distances(:,c) = sum( ...
            (Zte-centroids(c,:)).^2,2);
    end

    [~,idx] = min(distances,[],2);
    ypred = classes(idx);

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

        if predicted>0
            precision(c) = TP/predicted;
        end

        if actual>0
            recall(c) = TP/actual;
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

    fprintf('\n===== TRAIN %s -> TEST %s =====\n', ...
        trSession,teSession);

    fprintf('Training events: %d\n',sum(train));
    fprintf('Testing events: %d\n',sum(test));

    fprintf('\nConfusion Matrix:\n');

    disp(array2table(C, ...
        'VariableNames',{'Pred_D2','Pred_D3'}, ...
        'RowNames',{'Actual_D2','Actual_D3'}));

    fprintf('Accuracy: %.2f%%\n',accuracy);
    fprintf('Balanced Accuracy: %.2f%%\n',balanced);

    disp(table(classes,precision,recall,F1Score, ...
        'VariableNames', ...
        {'Device','Precision','Recall','F1Score'}));

    %% Save evaluation summary

    result = table( ...
        trSession,teSession,sum(train),sum(test), ...
        accuracy,balanced,recall(1),recall(2), ...
        'VariableNames', ...
        {'TrainSession','TestSession', ...
         'TrainCount','TestCount', ...
         'AccuracyPercent','BalancedAccuracyPercent', ...
         'D2Recall','D3Recall'});

    summaryResults = [summaryResults; result];

end

writetable(summaryResults,fullfile(outDir, ...
    'model01_ncc_cross_session_5f.csv'));

fprintf('\n===== FINAL SUMMARY =====\n');
disp(summaryResults);

fprintf('\nResults saved successfully.\n');
fprintf('===== MODEL 01 COMPLETE =====\n');
