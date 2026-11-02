
%% FINAL FOUR-MODEL PERFORMANCE COMPARISON
% Values transcribed from completed MATLAB experiments

clearvars; clc;

root = fileparts(mfilename('fullpath'));
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

models = ["NCC"; "RBF-SVM"; ...
    "Random Forest"; "k-NN"];

protocols = {
    'Within S01'
    'Within S02'
    'S01 to S02'
    'S02 to S01'
    'LOCO'
    };

% Rows: models
% Columns: evaluation protocols

A = [
    100, 98.92, 33.33, 35.02, 30.71;
    100, 100,   33.33, 35.02, 65.68;
    100, 100,   33.33, 34.34, 65.52;
    100, 100,   33.33, 35.35, 65.85
    ];

%% Export consolidated table

results = array2table(A, ...
    'VariableNames', ...
    {'WithinS01','WithinS02', ...
    'S01_to_S02','S02_to_S01','LOCO'});

results = addvars(results,models, ...
    'Before',1, ...
    'NewVariableNames','Classifier');

writetable(results,fullfile(outDir, ...
    'final_four_model_comparison.csv'));

%% Generate publication-ready grouped bar chart

figure('Color','w', ...
    'Position',[100 100 1150 600]);

bar(A','grouped');

ylabel('Classification Accuracy (%)');
title('RF Fingerprinting: Classifier Comparison');

xticks(1:5);
xticklabels(protocols);

ylim([0 110]);
grid on;

legend(cellstr(models), ...
    'Location','northoutside', ...
    'Orientation','horizontal');

set(gca,'FontSize',11);

%% Save high-resolution figure

exportgraphics(gcf,fullfile(outDir, ...
    'final_model_comparison.png'), ...
    'Resolution',300);

exportgraphics(gcf,fullfile(outDir, ...
    'final_model_comparison.pdf'), ...
    'ContentType','vector');

fprintf('\n===== FINAL COMPARISON =====\n');
disp(results);

fprintf('\nComparison CSV and figures saved.\n');
