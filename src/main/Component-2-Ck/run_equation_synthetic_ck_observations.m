%% Add path and directory
addpath(genpath("..\..\..\src\lib\utility\"));
addpath(genpath("..\..\..\models\Component-1-Lx"));
addpath(genpath("..\..\..\data\input\Component-2-Ck\Synthetic-Ck-Observations-Generation"));
addpath(genpath("..\..\..\data\output\synthetic-ck\"));
addpath(genpath("..\Component-3-Qul\"));

%% Read dataset
features_dataset = import_dataset("Features-Synthetic-Ck.xlsx", 10, "A2:J1462", "Sheet1", ...
    ["Date","Qriver","Qll", "Qtide", "Sll", "Socean", "h", "Ly", "utide", "Sul"], ...
    ["datetime", "double", "double", "double", "double", "double", "double", "double", "double", "double"]);

%% Load ML model to predict Lx
load("Component-1-Lx-Models.mat");
RF_model_Lx = component_1_trained_models.RF.model;
LSBoost_model_Lx = component_1_trained_models.LSBoost.model;

%% Compute Lx prediction
rf_pred = RF_model_Lx.predictFcn(features_dataset(:,["Qll","Qriver","Qtide","Sll"]));
lsboost_pred = LSBoost_model_Lx.predictFcn(features_dataset(:,["Qll","Qriver","Qtide","Sll"]));
features_dataset.Lx_RF_Pred = rf_pred;
features_dataset.Lx_LSBoost_Pred = lsboost_pred;

for i = 1:height(features_dataset)
    features_dataset.Qul(i) = compute_qul(features_dataset.Qriver(i), ...
        features_dataset.Qll(i), features_dataset.Qtide(i));

    features_dataset.Ck_RF(i) = generate_synthetic_ck( ...
        features_dataset.Sul(i),...
        features_dataset.Qul(i),...
        features_dataset.Lx_RF_Pred(i), ...
        features_dataset.h(i), ...
        features_dataset.Ly(i),...
        features_dataset.Socean(i),...
        features_dataset.utide(i),...
        features_dataset.Qll(i),...
        features_dataset.Sll(i));
    
    features_dataset.Ck_LSBoost(i) = generate_synthetic_ck(...
        features_dataset.Sul(i),...
        features_dataset.Qul(i),...
        features_dataset.Lx_LSBoost_Pred(i), ...
        features_dataset.h(i), ...
        features_dataset.Ly(i),...
        features_dataset.Socean(i),...
        features_dataset.utide(i),...
        features_dataset.Qll(i),...
        features_dataset.Sll(i));    
end

%% Remove missed data
idx_missed_data = features_dataset.Qriver == -999 | features_dataset.Sul==-999;
features_dataset.Lx_RF_Pred(idx_missed_data) = -999;
features_dataset.Lx_LSBoost_Pred(idx_missed_data) = -999;
features_dataset.Ck_RF(idx_missed_data) = -999;
features_dataset.Ck_LSBoost(idx_missed_data) = -999;
idx_missed_data = features_dataset.Qriver == -999;
features_dataset.Qul(idx_missed_data) = -999;

%% save dataset
writetable(features_dataset,"..\..\..\data\output\synthetic-ck\Synthetic-Ck-Obs.xlsx", "WriteRowNames", true);