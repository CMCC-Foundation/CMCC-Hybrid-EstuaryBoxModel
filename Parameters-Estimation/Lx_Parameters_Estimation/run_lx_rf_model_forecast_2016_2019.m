%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('1-Trained-Models\Trained-Test-Results-k-5-old-model-configuration'));

%% Set import dataset settings
filepath = "0-Dataset\Lx_forecast_2016_2019_CMEMS.xlsx";
nVars = 6;
dataRange = "A2:F1462";
sheetName = "Lx_2016_2019";
varNames = ["Year","Doy","Qll", "Qriver", "Sll", "Qtide"]; 
varTypes = ["int16","int16", "double", "double", "double", "double"];

[lx_dataset_2016_2019] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);
idx_missing_value = ckdatasetforhybridmodel.Dataset == 'Missed';
lx_dataset_2016_2019(idx_missing_value,:) = [];

%% Load trained model
load("1-Trained-Models\Trained-Test-Results-k-5-old-model-configuration\Trained-Tested-model-k-5.mat");

%% Predict on 2016-2019 data
random_forest_model = result_trained_model.RF.model;
lx_dataset_2016_2019.RF_Predictions = random_forest_model.predictFcn(lx_dataset_2016_2019);

%% Save results
writetable(lx_dataset_2016_2019,"0-Dataset\Lx_features_2016_2019_CMEMS_forecast_rf_predictions.xlsx");