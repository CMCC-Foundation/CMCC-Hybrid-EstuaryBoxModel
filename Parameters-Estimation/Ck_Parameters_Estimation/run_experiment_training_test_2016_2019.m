%% The following script group all the experiment carried out in this paper:
%  Given the dataset, we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 5. 
%  We use 80% of examples to train and validate our model, and 20% examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset\training_test_2016_2019_new_cal'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1_Trained-Models\'));

%% Set import dataset settings
% load ck-rf-obs training dataset

filepath = "0-Dataset\Ck-obs-by-rf\ck-training-dataset-rf.xlsx";
nVars = 7;
dataRange = "A2:G964";
sheetName = "ck_train";
varNames = ["Year","Doy","Qriver", "Qtide", "Socean", "CkObs","Dataset"]; 
varTypes = ["int16","int16","double", "double","double","double","categorical"];
[ck_training_dataset_rf] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%load ck-rf-obs test dataset
filepath = "0-Dataset\Ck-obs-by-rf\ck-test-dataset-rf.xlsx";
dataRange = "A2:G239";
sheetName = "ck_test";
[ck_test_dataset_rf] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%load ck-lsboost-obs training dataset
filepath = "0-Dataset\Ck-obs-by-lsboost\ck-training-dataset-lsboost.xlsx";
dataRange = "A2:G964";
sheetName = "ck_train";
[ck_training_dataset_lsboost] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%load ck-rf-obs test dataset
filepath = "0-Dataset\Ck-obs-by-lsboost\ck-test-dataset-lsboost.xlsx";
dataRange = "A2:G239";
sheetName = "ck_test";
[ck_test_dataset_lsboost] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

pwbTable = table('Size',[numel(pwbX) numel(algorithm_names)],...
    'VariableTypes', repmat({'double'}, 1, numel(algorithm_names)), ...
    'VariableNames', algorithm_names,...
    'RowNames', pwbXRowNames);

results_training = table('Size', [numel(algorithm_names) 8], ...
    'VariableTypes', {'double','double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [numel(algorithm_names) 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

%% Set target feature for the machine and deep learning model
targetFeatureName = 'CkObs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 100;

%% Set k to be use in k-fold cross validation
k = 5;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function( ...
    ck_training_dataset_rf(:,["Qriver","Qtide","Socean","CkObs"]), ...
    targetFeatureName, ...
    max_objective_evaluations, ...
    k);

result_trained_model.random_forest.validation_results.validation_predictions...
    (result_trained_model.random_forest.validation_results.validation_predictions < 0) = 0;

results_training = compute_metrics( ...
    ck_training_dataset_rf.CkObs,...
    result_trained_model.random_forest.validation_results.validation_predictions, ...
    algorithm_names(1), ...
    results_training ...
    );

result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for ck estimation with Random Forest");

% save test results
test_results = struct();
result_trained_model.random_forest.test_results = test_results;
result_trained_model.random_forest.test_results.test_predictions = ...
    result_trained_model.random_forest.model.predictFcn(ck_test_dataset_rf(:,["Qriver","Qtide","Socean"]));

result_trained_model.random_forest.test_results.test_predictions...
    (result_trained_model.random_forest.test_results.test_predictions < 0) = 0;

results_test= compute_metrics( ...
    ck_test_dataset_rf.CkObs, ...
    result_trained_model.random_forest.test_results.test_predictions, ...
    algorithm_names(1), ...
    results_test);

result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);
pwbTable = create_pwb_table( ...
    ck_test_dataset_rf.CkObs, ...
    result_trained_model.random_forest.test_results.test_predictions, ...
    pwbTable, ...
    algorithm_names(1), ...
    pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.lsboost = lsboost_function( ...
    ck_training_dataset_lsboost(:,["Qriver","Qtide","Socean","CkObs"]), ...
    targetFeatureName, ...
    max_objective_evaluations, ...
    k);

result_trained_model.lsboost.validation_results.validation_predictions...
    (result_trained_model.lsboost.validation_results.validation_predictions < 0) = 0;

results_training = compute_metrics( ...
    ck_training_dataset_lsboost.CkObs, ...
    result_trained_model.lsboost.validation_results.validation_predictions, ...
    algorithm_names(2), ...
    results_training);

result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for ck estimation with Lsboost");

% save test results
test_results = struct();
result_trained_model.lsboost.test_results = test_results;
result_trained_model.lsboost.test_results.test_predictions = ...
    result_trained_model.lsboost.model.predictFcn(ck_test_dataset_lsboost(:,["Qriver","Qtide","Socean"]));

result_trained_model.lsboost.test_results.test_predictions...
    (result_trained_model.lsboost.test_results.test_predictions < 0) = 0;

results_test= compute_metrics( ...
    ck_test_dataset_lsboost.CkObs, ...
    result_trained_model.lsboost.test_results.test_predictions, ...
    algorithm_names(2), ...
    results_test);

result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);
pwbTable = create_pwb_table( ...
    ck_test_dataset_lsboost.CkObs, ...
    result_trained_model.lsboost.test_results.test_predictions, ...
    pwbTable, ...
    algorithm_names(2), ...
    pwbX);

%% Display results
clc;
disp(results_training(:,["RMSE","MAE","Corr Coeff", "R2"]));
disp(results_test(:,["RMSE","MAE","Corr Coeff","R2"]));
disp(pwbTable);

%writetable(results_training, '1-Trained-Models/Results-Ck-training-model.xlsx', 'WriteRowNames',true);
%writetable(results_test, '1-Trained-Models/Results-Ck-test-model.xlsx', 'WriteRowNames',true);
%writetable(pwbTable, "1-Trained-Models/pwbTable.xlsx", "WriteRowNames", true);
%save("1-Trained-Models\Ck-Trained-Tested-model-new-cal.mat","result_trained_model");