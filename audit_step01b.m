
%% STEP 01B - Quality and Event Integrity
% Read only. Does not modify the original CSV.

clearvars;
clc;

csvPath = 'D:\Project\master_features_accepted.csv';

T = readtable(csvPath, ...
    'VariableNamingRule', 'preserve');

%% 1. Quality status
fprintf('\n===== QUALITY STATUS =====\n');

status = strtrim(string(T.quality_status));
reasons = strtrim(string(T.quality_reasons));

[statusNames, ~, group] = unique(status);
counts = accumarray(group, 1);

disp(table(statusNames, counts, ...
    'VariableNames', {'Status','Rows'}));

blankReasons = ismissing(reasons) | reasons == "";
fprintf('Blank quality reasons: %d\n', sum(blankReasons));
fprintf('Nonblank quality reasons: %d\n', sum(~blankReasons));

%% 2. Record ID duplicates
fprintf('\n===== RECORD ID CHECK =====\n');

recordIDs = string(T.record_id);

fprintf('Duplicate record IDs: %d\n', ...
    numel(recordIDs) - numel(unique(recordIDs)));

%% 3. Event identity duplicates
fprintf('\n===== EVENT ID CHECK =====\n');

device = strtrim(string(T.device_id));
session = strtrim(string(T.session_id));
capture = strtrim(string(T.capture_id));
eventIndex = T.event_index_1based;

eventKeys = device + "/" + session + "/" + ...
    capture + "/" + string(eventIndex);

fprintf('Duplicate device/session/capture/event keys: %d\n', ...
    numel(eventKeys) - numel(unique(eventKeys)));

fprintf('Invalid/non-integer event indices: %d\n', ...
    sum(~isfinite(eventIndex) | ...
    eventIndex < 1 | eventIndex ~= floor(eventIndex)));

%% 4. Event index ranges by capture
fprintf('\n===== EVENT INDEX RANGES =====\n');

[G, d, s, c] = findgroups(device, session, capture);

rows = splitapply(@numel, eventIndex, G);
firstEvent = splitapply(@min, eventIndex, G);
lastEvent = splitapply(@max, eventIndex, G);

disp(table(d, s, c, rows, firstEvent, lastEvent, ...
    'VariableNames', ...
    {'Device','Session','Capture','Rows', ...
    'MinEventIndex','MaxEventIndex'}));

fprintf('\n===== STEP 01B COMPLETE =====\n');
