
%% CAPTURE-SEPARATED RF CLASSIFICATION
% Leave-one-capture-out (LOCO)
% Uses F2-F5 and nearest-centroid classification
% Training-only standardization

clearvars; clc;

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

keys = string(T.device_id) + "/" + ...
       string(T.session_id) + "/" + ...
       string(T.capture_id);

classes = ["D2";"D3"];

eligible = string(T.quality_status)=="ACCEPT" & ...
    ismember(y,classes) & all(isfinite(X),2);

folds = unique(keys(eligible),'stable');

pred = strings(height(T),1);

foldName = strings(numel(folds),1);
nTest = zeros(numel(folds),1);
nCorrect = zeros(numel(folds),1);
foldAccuracy = zeros(numel(folds),1);

fprintf('\n===== LEAVE-ONE-CAPTURE-OUT =====\n');

for f = 1:numel(folds)

    test = eligible & keys==folds(f);
    train = eligible & keys~=folds(f);

    Xtr = X(train,:);
    Xte = X(test,:);

    ytr = y(train);

    % Training-only normalization
    mu = mean(Xtr,1);
    sd = std(Xtr,0,1);
    sd(sd==0)=1;

    Ztr = (Xtr-mu)./sd;
    Zte = (Xte-mu)./sd;

    % Nearest-centroid classifier
    centroid = zeros(2,size(Ztr,2));

    for c = 1:2
        assert(any(ytr==classes(c)), ...
            'Missing training class.');
        centroid(c,:) = ...
            mean(Ztr(ytr==classes(c),:),1);
    end

    dist = zeros(size(Zte,1),2);

    for c = 1:2
        dist(:,c) = sum( ...
            (Zte-centroid(c,:)).^2,2);
    end

    [~,ix] = min(dist,[],2);
    prediction = classes(ix);

    pred(test)=prediction;

    foldName(f)=folds(f);
    nTest(f)=sum(test);
    nCorrect(f)=sum(prediction==y(test));
    foldAccuracy(f)=100*nCorrect(f)/nTest(f);

    fprintf('%s: %.2f%% (%d/%d)\n', ...
        char(folds(f)),foldAccuracy(f), ...
        nCorrect(f),nTest(f));
end

%% Aggregate confusion matrix

C=zeros(2,2);

for i=1:2
    for j=1:2
        C(i,j)=sum(eligible & ...
            y==classes(i) & pred==classes(j));
    end
end

fprintf('\n===== AGGREGATE CONFUSION MATRIX =====\n');

disp(array2table(C, ...
    'VariableNames',{'Pred_D2','Pred_D3'}, ...
    'RowNames',{'Actual_D2','Actual_D3'}));

accuracy = 100*sum(diag(C))/sum(C(:));

precision=nan(2,1);
recall=nan(2,1);
f1=nan(2,1);

for c=1:2
    tp=C(c,c);
    actual=sum(C(c,:));
    predicted=sum(C(:,c));

    if actual>0
        recall(c)=tp/actual;
    end
    if predicted>0
        precision(c)=tp/predicted;
    end
    if isfinite(precision(c)) && ...
       precision(c)+recall(c)>0
        f1(c)=2*precision(c)*recall(c)/ ...
              (precision(c)+recall(c));
    elseif actual>0 && predicted==0
        f1(c)=0;
    end
end

fprintf('\n===== OVERALL METRICS =====\n');
fprintf('Accuracy: %.2f%%\n',accuracy);
fprintf('Balanced accuracy: %.2f%%\n', ...
    100*mean(recall));

disp(table(classes,precision,recall,f1, ...
    'VariableNames', ...
    {'Device','Precision','Recall','F1Score'}));

%% Save derived results

foldTable=table(foldName,nTest,nCorrect,foldAccuracy);

writetable(foldTable,fullfile(outDir, ...
    'capture_loco_results.csv'));

predictionTable=table( ...
    string(T.record_id(eligible)), ...
    keys(eligible),y(eligible),pred(eligible), ...
    'VariableNames', ...
    {'RecordID','Capture','Actual','Predicted'});

writetable(predictionTable,fullfile(outDir, ...
    'capture_loco_predictions.csv'));

fprintf('\nResults saved to Results folder.\n');
fprintf('===== EVALUATION COMPLETE =====\n');
