
%% STEP 01 - Dataset and Capture Coverage Audit
% Read-only audit. No source files are modified.

clearvars;
clc;

%% 1. Locate original data

projectDir = 'D:\Project';

csvPath = fullfile(projectDir, ...
    'master_features_accepted.csv');

matRoot = fullfile(projectDir, 'RF_Dataset');

assert(isfile(csvPath), 'CSV not found: %s', csvPath);
assert(isfolder(matRoot), 'MAT folder not found: %s', matRoot);

fprintf('CSV found successfully.\n');
fprintf('RF_Dataset found successfully.\n');

%% 2. Load CSV and inspect schema
opts = detectImportOptions(csvPath, ...
    'VariableNamingRule', 'preserve');
T = readtable(csvPath, opts);

fprintf('\n===== CSV OVERVIEW =====\n');
fprintf('Total rows: %d\n', height(T));
fprintf('Total columns: %d\n', width(T));

names = string(T.Properties.VariableNames(:));
missingCount = zeros(width(T), 1);
nonFiniteCount = zeros(width(T), 1);
types = strings(width(T), 1);

for k = 1:width(T)
    v = T.(T.Properties.VariableNames{k});
    types(k) = string(class(v));
    missingCount(k) = sum(ismissing(v), 'all');

    if isnumeric(v)
        nonFiniteCount(k) = sum(~isfinite(v), 'all');
    end
end

disp(table(names, types, missingCount, nonFiniteCount, ...
    'VariableNames', ...
    {'Column', 'Type', 'Missing', 'NonFinite'}));

%% 3. Device/session/capture coverage
required = ["device_id", "session_id", "capture_id"];
assert(all(ismember(required, names)), ...
    'Required identifier columns missing.');

dev = upper(strtrim(string(T.device_id)));
ses = upper(strtrim(string(T.session_id)));
cap = upper(strtrim(string(T.capture_id)));

validID = ~ismissing(dev) & ~ismissing(ses) & ...
          ~ismissing(cap) & ...
          strlength(dev)>0 & strlength(ses)>0 & ...
          strlength(cap)>0;

fprintf('\n===== CSV CAPTURE COUNTS =====\n');
fprintf('Rows with invalid/missing IDs: %d\n', sum(~validID));

[G, d, s, c] = findgroups( ...
    dev(validID), ses(validID), cap(validID));

n = splitapply(@numel, dev(validID), G);

coverage = table(d, s, c, n, ...
    'VariableNames', ...
    {'Device', 'Session', 'Capture', 'Rows'});

disp(sortrows(coverage, {'Device','Session','Capture'}));

fprintf('D2 S02 rows: %d\n', ...
    sum(dev=="D2" & ses=="S02"));
fprintf('D3 S02 rows: %d\n', ...
    sum(dev=="D3" & ses=="S02"));

%% 4. Exact duplicate rows
fprintf('\n===== DUPLICATE AUDIT =====\n');

try
    [~, firstRows] = unique(T, 'rows', 'stable');
    duplicateRows = setdiff((1:height(T))', firstRows);

    fprintf('Exact duplicate rows: %d\n', ...
        numel(duplicateRows));
    fprintf('First duplicate row indices:\n');
    disp(duplicateRows(1:min(10,numel(duplicateRows))));
catch ME
    fprintf('Duplicate check could not run: %s\n', ME.message);
end

% Repeated device/session/capture labels are expected
% because each capture can contain many burst events.

%% 5. Quality flag distribution
fprintf('\n===== QUALITY FLAGS =====\n');

if ismember("quality_flag", names)
    flags = string(T.quality_flag);
    [gf, flagNames] = findgroups(flags);
    flagCounts = splitapply(@numel, flags, gf);

    disp(table(flagNames, flagCounts, ...
        'VariableNames', {'QualityFlag','Rows'}));
    fprintf('Missing quality flags: %d\n', ...
        sum(ismissing(flags)));
else
    fprintf('quality_flag column not found.\n');
end

%% 6. MAT source-file inventory (no MAT files loaded)
fprintf('\n===== SOURCE MAT INVENTORY =====\n');

files = dir(fullfile(matRoot, '**', '*.mat'));
fprintf('Total .mat files found: %d\n', numel(files));

md = strings(numel(files),1);
ms = strings(numel(files),1);
mc = strings(numel(files),1);

for k = 1:numel(files)
    path = fullfile(files(k).folder, files(k).name);
    rel = upper(erase(path, [matRoot filesep]));

    a = regexp(rel, 'D[0-9]+', 'match', 'once');
    b = regexp(rel, 'S[0-9]+', 'match', 'once');
    z = regexp(rel, 'C[0-9]+', 'match', 'once');

    if ~isempty(a), md(k) = string(a); end
    if ~isempty(b), ms(k) = string(b); end
    if ~isempty(z), mc(k) = string(z); end
end

parsed = strlength(md)>0 & strlength(ms)>0 & ...
         strlength(mc)>0;

fprintf('Files with identifiable D/S/C: %d\n', sum(parsed));
fprintf('Unidentified files: %d\n', sum(~parsed));

if any(~parsed)
    fprintf('Unidentified paths:\n');
    for k = find(~parsed)'
        disp(fullfile(files(k).folder, files(k).name));
    end
end

if any(parsed)
    [gm, dm, sm, cm] = findgroups( ...
        md(parsed), ms(parsed), mc(parsed));

    nm = splitapply(@numel, md(parsed), gm);

    matCoverage = table(dm, sm, cm, nm, ...
        'VariableNames', ...
        {'Device','Session','Capture','MAT_Files'});

    disp(sortrows(matCoverage, ...
        {'Device','Session','Capture'}));
end

%% 7. Compare capture identifiers across sources
fprintf('\n===== CSV VS MAT COVERAGE =====\n');

csvKeys = unique(dev(validID) + "/" + ...
    ses(validID) + "/" + cap(validID));

matKeys = unique(md(parsed) + "/" + ...
    ms(parsed) + "/" + mc(parsed));

fprintf('CSV captures without a matched MAT identifier:\n');
disp(setdiff(csvKeys, matKeys));

fprintf('MAT captures without a matched CSV identifier:\n');
disp(setdiff(matKeys, csvKeys));

fprintf('\n===== STEP 01 COMPLETE =====\n');
