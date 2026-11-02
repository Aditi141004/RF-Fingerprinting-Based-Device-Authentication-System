
%% MODEL 01 - NEAREST CENTROID CLASSIFIER
% Five RF features: F1, F2, F3, F4, F5
% Within-session 70/30 burst-level split
% Training-only Z-score normalization
% Fixed random seed for reproducibility

clearvars; clc;
rng(42);

root = 'D:\Project';
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'));

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

captureKey = y + "/" + session + "/" + ...
    string(T.capture_id);

classes = ["D2";"D3"];

splitRole = repmat("EXCLUDED",height(T),1);

fprintf('\n===== MODEL 01: NCC (F1-F5) =====\n');

for ss = ["S01","S02"]

    eligible = session==ss & status=="ACCEPT" & ...
        ismember(y,classes) & all(isfinite(X),2);

    captures = unique(captureKey(eligible));

    train = false(height(T),1);
    test = false(height(T),1);

    %% Split each capture into 70% train, 30% test

    for k = 1:numel(captures)

        idx = find(eligible & captureKey==captures(k));
        order = idx(randperm(numel(idx)));

        nTest = max(1,round(0.30*numel(idx)));

        test(order(1:nTest)) = true;
        train(order(nTest+1:end)) = true;
    end

    splitRole(train) = "TRAIN";
    splitRole(test) = "TEST";

    %% Training-only normalization

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);
    yte = y(test);

    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0) = 1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    %% Train nearest-centroid classifier

    centroids = zeros(2,size(X,2));

    for c = 1:2
        centroids(c,:) = ...
            mean(Ztr(ytr==classes(c),:),1);
    end

    %% Predict device identities

    D = zeros(size(Zte,1),2);

    for c = 1:2
        D(:,c) = sum( ...
            (Zte-centroids(c,:)).^2,2);
    end

    [~,idx] = min(D,[],2);
    ypred = classes(idx);

    %% Confusion matrix

    C = zeros(2,2);

    for i = 1:2
        for j = 1:2
            C(i,j) = sum(yte==classes(i) & ...
                         ypred==classes(j));
        end
    end

    fprintf('\n===== SESSION %s =====\n',ss);
    fprintf('Training events: %d\n',sum(train));
    fprintf('Testing events: %d\n',sum(test));

    disp(array2table(C, ...
        'VariableNames',{'Pred_D2','Pred_D3'}, ...
        'RowNames',{'Actual_D2','Actual_D3'}));

    %% Metrics

    accuracy = 100*sum(diag(C))/sum(C(:));

    precision = zeros(2,1);
    recall = zeros(2,1);
    F1score = zeros(2,1);

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

        if precision(c)+recall(c)>0
            F1score(c) = ...
                2*precision(c)*recall(c) / ...
                (precision(c)+recall(c));
        end
    end

    fprintf('Accuracy: %.2f%%\n',accuracy);
    fprintf('Balanced accuracy: %.2f%%\n', ...
        100*mean(recall));

    disp(table(classes,precision,recall,F1score, ...
        'VariableNames', ...
        {'Device','Precision','Recall','F1Score'}));
end

%% Save partitions for other ML models

splitTable = table( ...
    string(T.record_id),captureKey,splitRole, ...
    'VariableNames', ...
    {'RecordID','Capture','Partition'});

writetable(splitTable,fullfile(outDir, ...
    'within_session_split_rng42.csv'));

fprintf('\nSaved fixed train/test partitions.\n');
fprintf('===== MODEL 01 COMPLETE =====\n');
