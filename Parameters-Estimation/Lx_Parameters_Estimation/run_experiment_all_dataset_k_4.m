%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 4. 
%  In the end, some figures are plotted and dataset and model are saved. 

%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('1-Trained-Models'));

%% Set import dataset settings
filepath = "0-Dataset\LX_OBS_WITH_FEATURES.xlsx";
nVars = 7;
dataRange = "A2:G26";
sheetName = "Lx_obs";
varNames = ["DATE","Q_l", "Q_r", "S_l", "Q_tide", "Lx_OBS", "Dataset_Type"]; 
varTypes = ["datetime", "double", "double", "double", "double","double","categorical"];

[lx_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);
save('0-Dataset/LX_OBS_WITH_FEATURES.mat', ...
    'lx_dataset');

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Lx_OBS';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 60;

%% Removed useless features
training_dataset = lx_dataset;

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost' };

results = table('Size', [2 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

k = 4;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.random_forest = random_forest_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
results = compute_metrics(training_dataset(:,targetFeatureName),result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results);
result_trained_model.random_forest.validation_results.metrics = results("random_forest",:);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.lsboost = lsboost_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
results = compute_metrics(training_dataset(:,targetFeatureName),result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results);
result_trained_model.lsboost.validation_results.metrics = results("lsboost",:);

clc;
close all;

writetable(results, '1-Trained-Models/Trained-All-Dataset-k-4/Results-Trained-model-k-4.xlsx', 'WriteRowNames',true);
save("1-Trained-Models/Trained-All-Dataset-k-4/Trained-model-k-4.mat","result_trained_model");