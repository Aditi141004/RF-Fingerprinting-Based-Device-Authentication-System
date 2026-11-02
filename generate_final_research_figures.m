
%% FINAL RF FINGERPRINTING RESEARCH FIGURES
% Based on verified classification counts from Models 01-04
% Exact percentages computed from correct/total events
% Generates two high-resolution report-ready figures

clearvars; close all; clc;

root = fileparts(mfilename('fullpath'));
outDir = fullfile(root,'Results');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% =========================================================
%  1. EXPERIMENTAL DATA
% ==========================================================

models = {'NCC','RBF-SVM','Random Forest','k-NN'};

protocols = {
    'Within S01'
    'Within S02'
    'S01 -> S02'
    'S02 -> S01'
    'LOCO'
};

% Rows = evaluation protocols
% Columns = NCC, SVM, RF, KNN
% Each entry is the number of CORRECT predictions.

correct = [
     89   89   89   89;
     92   93   93   93;
    104  104  104  104;
    104  104  102  105;
    187  400  399  401
];

% Total held-out events per evaluation

total = [89; 93; 312; 297; 609];

% Compute accuracy directly from integer counts

accuracy = 100 * correct ./ total;

fprintf('\n===== EXACT ACCURACY VALUES =====\n');

disp(array2table(accuracy, ...
    'VariableNames', ...
    {'NCC','RBF_SVM','RandomForest','KNN'}, ...
    'RowNames', ...
    {'WithinS01','WithinS02', ...
     'S01_to_S02','S02_to_S01','LOCO'}));

%% =========================================================
%  2. FIGURE 1 - COMPLETE MODEL COMPARISON
% ==========================================================

fig1 = figure( ...
    'Color','w', ...
    'Position',[80 80 1450 680]);

ax1 = axes(fig1);

b = bar(ax1,accuracy,'grouped');
hold(ax1,'on');

% Distinct, high-contrast colors

colors = [
    0.12 0.36 0.66;
    0.78 0.30 0.14;
    0.31 0.53 0.22;
    0.45 0.22 0.67
];

for j = 1:4
    b(j).FaceColor = colors(j,:);
    b(j).EdgeColor = 'none';
end

% Exact percentages inside each bar
% Vertical labels avoid overlap in grouped bars

for j = 1:4

    xpos = b(j).XEndPoints;
    ypos = b(j).YEndPoints;

    labels = compose('%.2f%%',accuracy(:,j));

    text(ax1,xpos,ypos-1.5,labels, ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','middle', ...
        'Rotation',90, ...
        'FontSize',9, ...
        'FontWeight','bold', ...
        'Color','w');
end

% Axis configuration

ylim(ax1,[0 100]);
yticks(ax1,0:10:100);

xticks(ax1,1:5);
xticklabels(ax1,protocols);

ylabel(ax1,'Classification Accuracy (%)', ...
    'FontWeight','bold');

xlabel(ax1,'Evaluation Protocol', ...
    'FontWeight','bold');

title(ax1, ...
    'RF Fingerprinting: Performance Across Evaluation Protocols', ...
    'FontSize',15,'FontWeight','bold');

subtitle(ax1, ...
    '609 accepted events | 2 known devices | 5 RF fingerprint features');

legend(ax1,b,models, ...
    'Location','northoutside', ...
    'Orientation','horizontal', ...
    'NumColumns',4);

grid(ax1,'on');
ax1.YGrid = 'on';
ax1.XGrid = 'off';
ax1.GridAlpha = 0.15;
ax1.FontSize = 11;
ax1.LineWidth = 1;
box(ax1,'off');

%% Export Figure 1

exportgraphics(fig1, ...
    fullfile(outDir,'final_comparison_improved.png'), ...
    'Resolution',300);

exportgraphics(fig1, ...
    fullfile(outDir,'final_comparison_improved.pdf'), ...
    'ContentType','vector');

%% =========================================================
%  3. FIGURE 2 - LOCO CAPTURE-BY-CAPTURE HEATMAP
% ==========================================================

captures = {
    'D2/S01/C01'
    'D2/S01/C02'
    'D2/S02/C02'
    'D3/S01/C01'
    'D3/S02/C01'
    'D3/S02/C02'
};

% Rows = six held-out captures
% Columns = four classifiers
% Correct counts from completed LOCO experiments

locoCorrect = [
     83   88   88   89;
    104  104  104  104;
      0    0    0    0;
      0    0    0    0;
      0  104  104  104;
      0  104  103  104
];

locoTotal = [89;104;104;104;104;104];

locoAccuracy = 100 * locoCorrect ./ locoTotal;

% Validate totals against aggregate LOCO counts

assert(isequal(sum(locoCorrect,1),correct(5,:)), ...
    'LOCO fold counts do not match aggregate results.');

assert(sum(locoTotal)==609, ...
    'Expected 609 LOCO events.');

fig2 = figure( ...
    'Color','w', ...
    'Position',[100 100 1120 620]);

ax2 = axes(fig2);

imagesc(ax2,locoAccuracy);

colormap(ax2,parula(256));
clim(ax2,[0 100]);

cb = colorbar(ax2);
cb.Label.String = 'Capture Classification Accuracy (%)';

xticks(ax2,1:4);
xticklabels(ax2,models);

yticks(ax2,1:6);
yticklabels(ax2,captures);

xlabel(ax2,'Classifier','FontWeight','bold');
ylabel(ax2,'Held-Out Capture','FontWeight','bold');

title(ax2, ...
    'Leave-One-Capture-Out: Accuracy by Held-Out Capture', ...
    'FontSize',14,'FontWeight','bold');

subtitle(ax2, ...
    'Six folds | 609 test events in aggregate');

ax2.FontSize = 11;
ax2.TickLength = [0 0];

% Annotate every cell

for r = 1:6
    for c = 1:4

        value = locoAccuracy(r,c);

        % Adjust text contrast
        if value < 50
            txtColor = 'w';
        else
            txtColor = 'k';
        end

        text(ax2,c,r,sprintf('%.2f%%',value), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontWeight','bold', ...
            'FontSize',11, ...
            'Color',txtColor);
    end
end

% White grid boundaries between cells

hold(ax2,'on');

for x = 0.5:1:4.5
    xline(ax2,x,'w','LineWidth',1.5);
end

for y = 0.5:1:6.5
    yline(ax2,y,'w','LineWidth',1.5);
end

%% Export Figure 2

exportgraphics(fig2, ...
    fullfile(outDir,'loco_capture_heatmap.png'), ...
    'Resolution',300);

exportgraphics(fig2, ...
    fullfile(outDir,'loco_capture_heatmap.pdf'), ...
    'ContentType','vector');

%% =========================================================
%  4. FINAL OUTPUT
% ==========================================================

fprintf('\n===== EXACT LOCO RESULTS =====\n');

for j = 1:4
    fprintf('%-15s : %d/609 = %.4f%%\n', ...
        models{j},correct(5,j),accuracy(5,j));
end

fprintf('\n===== FIGURES SAVED =====\n');
fprintf('1. final_comparison_improved.png\n');
fprintf('2. final_comparison_improved.pdf\n');
fprintf('3. loco_capture_heatmap.png\n');
fprintf('4. loco_capture_heatmap.pdf\n');

fprintf('\n===== COMPLETE =====\n');
