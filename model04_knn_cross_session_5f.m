
%% MODEL 04 - KNN CROSS-SESSION EVALUATION
% Five features: F1-F5
% k = 5, Euclidean distance, equal voting
% Training-only normalization
% S01 -> S02 and S02 -> S01

clearvars; clc;

root = fileparts(mfilename('fullpath'));
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'), ...
    'VariableNamingRule','preserve');

%% Features

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

fprintf('\n===== MODEL 04: KNN CROSS-SESSION =====\n');

for fold = 1:2

    tr = trainSessions(fold);
    te = testSessions(fold);

    train = eligible & session==tr;
    test = eligible & session==te;

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    assert(all(ismember(classes,unique(ytr))), ...
        'Both classes required in training.');

    %% Training-only Z-score normalization

    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0)=1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    %% Train k-NN

    mdl = fitcknn(Ztr,categorical(ytr), ...
        'NumNeighbors',5, ...
        'Distance','euclidean', ...
        'DistanceWeight','equal', ...
        'Standardize',false);

    %% Predict held-out session

    ypred = string(predict(mdl,Zte));

    %% Confusion matrix

    C = zeros(2,2);

    for i = 1:2
        for j = 1:2
            C(i,j) = sum(yte==classes(i) & ...
                         ypred==classes(j));
        end
    end

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

    %% Display results

    fprintf('\n===== TRAIN %s -> TEST %s =====\n',tr,te);

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

    %% Save model and preprocessing parameters

    save(fullfile(outDir, ...
        sprintf('model04_knn_train_%s_5f.mat',char(tr))), ...
        'mdl','mu','sd','featureNames','classes');

    %% Store summary

    result = table(tr,te,sum(train),sum(test), ...
        accuracy,balanced,recall(1),recall(2), ...
        'VariableNames', ...
        {'TrainSession','TestSession', ...
         'TrainCount','TestCount', ...
         'AccuracyPercent','BalancedAccuracyPercent', ...
         'D2Recall','D3Recall'});

    summaryResults = [summaryResults;result];

end

%% Export results

writetable(summaryResults,fullfile(outDir, ...
    'model04_knn_cross_session_5f.csv'));

fprintf('\n===== MODEL 04 CROSS-SESSION SUMMARY =====\n');
disp(summaryResults);

fprintf('\n===== EVALUATION COMPLETE =====\n');
