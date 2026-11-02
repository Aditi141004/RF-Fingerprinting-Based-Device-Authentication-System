
%% FEATURE DISTRIBUTIONS - RF Fingerprinting
% Read-only source analysis; outputs saved separately

clearvars; clc;

root = 'D:\Project';
outDir = fullfile(root, 'Results');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

T = readtable(fullfile(root, ...
    'master_features_accepted.csv'));

names = {
    'F1_transient_energy'
    'F2_mean_amplitude'
    'F3_amplitude_variance'
    'F4_residual_phase_variance'
    'F5_tone_offset_hz'
};

labels = {
    'F1: Transient energy (stored scale)'
    'F2: Mean amplitude (IQ units)'
    'F3: Amplitude variance (IQ units squared)'
    'F4: Residual phase variance (rad squared)'
    'F5: Tone frequency offset (Hz)'
};

X = T{:,names};

captureKeys = string(T.device_id) + "/" + ...
              string(T.session_id) + "/" + ...
              string(T.capture_id);

[groups,~,groupIndex] = unique(captureKeys,'stable');

%% Calculate group medians

M = zeros(numel(groups),5);

for g = 1:numel(groups)
    M(g,:) = median(X(groupIndex==g,:),1);
end

summary = array2table(M, ...
    'VariableNames',{'F1','F2','F3','F4','F5'});

summary = addvars(summary, groups, ...
    'Before',1, 'NewVariableNames','Capture');

fprintf('\n===== FEATURE MEDIANS BY CAPTURE =====\n');
disp(summary);

writetable(summary,fullfile(outDir, ...
    'feature_medians.csv'));

%% Generate five-panel feature plot

figure('Color','w','Position',[50 50 1400 1100]);

tiledlayout(5,1,'TileSpacing','compact', ...
    'Padding','compact');

for f = 1:5

    nexttile;
    hold on;

    for g = 1:numel(groups)

        idx = find(groupIndex==g);
        y = X(idx,f);

        % Deterministic horizontal spreading
        offset = linspace(-0.20,0.20,numel(idx))';
        x = g + offset;

        scatter(x,y,9,'filled', ...
            'MarkerFaceAlpha',0.35);

        med = median(y);
        plot([g-0.25,g+0.25],[med,med], ...
            'k-','LineWidth',2);
    end

    if (f==3 || f==4) && all(X(:,f)>0)
        set(gca,'YScale','log');
    end

    ylabel(labels{f});
    xlim([0.5,numel(groups)+0.5]);
    xticks(1:numel(groups));
    xticklabels(groups);
    grid on;
    box on;
    hold off;
end

sgtitle('RF Fingerprint Feature Distributions by Capture');

exportgraphics(gcf,fullfile(outDir, ...
    'features_by_capture.png'),'Resolution',200);

fprintf('\nSaved results in: %s\n',outDir);
fprintf('===== PLOTTING COMPLETE =====\n');
