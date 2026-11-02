
%% STEP 02A - Missing Captures and Event Gaps
% Read-only inspection

clearvars;
clc;

projectDir = 'D:\Project';

%% PART A: Identify missing event indices

T = readtable(fullfile(projectDir, ...
    'master_features_accepted.csv'));

idx = strcmp(string(T.device_id), 'D2') & ...
    strcmp(string(T.session_id), 'S01') & ...
    strcmp(string(T.capture_id), 'C01');

events = sort(T.event_index_1based(idx));

missingEvents = setdiff(1:max(events), events);

fprintf('\n===== MISSING EVENT INDICES =====\n');
fprintf('D2 / S01 / C01\n');
fprintf('Events in CSV: %d\n', numel(events));
fprintf('Missing indices: ');
disp(missingEvents);

%% PART B: Inspect MAT variable structure

fprintf('\n===== MAT FILE STRUCTURES =====\n');

captures = {
    'D2','S02','C01.mat';
    'D3','S01','C02.mat';
    'D2','S01','C01.mat'
    };

for k = 1:size(captures,1)

    filePath = fullfile(projectDir, ...
        'RF_Dataset', ...
        captures{k,1}, ...
        captures{k,2}, ...
        captures{k,3});

    fprintf('\n--- %s / %s / %s ---\n', ...
        captures{k,1}, captures{k,2}, captures{k,3});

    if ~isfile(filePath)
        fprintf('FILE NOT FOUND\n');
        continue;
    end

    info = whos('-file', filePath);

    fprintf('Variables found: %d\n', numel(info));

    for j = 1:numel(info)
        fprintf('%-32s %-12s Size: %s\n', ...
            info(j).name, ...
            info(j).class, ...
            mat2str(info(j).size));
    end

end

fprintf('\n===== STEP 02A COMPLETE =====\n');
