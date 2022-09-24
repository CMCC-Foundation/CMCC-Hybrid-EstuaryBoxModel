%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 4. 
%  We use 20 examples to train and validate our model, and 5 examples to test it. 

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

%% Removed useless features and split dataset in training and test set
training_dataset = lx_dataset(strcmp(string(lx_dataset.Dataset_Type), "CALIBRATION RANGE"),:);
testing_dataset = lx_dataset(strcmp(string(lx_dataset.Dataset_Type), 'VALIDATION RANGE'),:);

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'old_model'};

results_training = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE','MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

k = 4;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(training_dataset(:,targetFeatureName),result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);

% save test results
result_trained_model.random_forest.test_results.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(testing_dataset, {'DATE','Dataset_Type'}));
results_test = compute_metrics(testing_dataset(:,targetFeatureName), result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.lsboost = lsboost_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(training_dataset(:,targetFeatureName),result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);

% save test results
result_trained_model.lsboost.test_results.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(testing_dataset, {'DATE','Dataset_Type'}));
results_test = compute_metrics(testing_dataset(:,targetFeatureName), result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);

%% Compute metrics on old model
load("0-Dataset\Old_Model_Training_Test_Prediction.mat");
old_model_results = struct();

% compute training performance
results_training = compute_metrics(Old_Model_Training.LX_Obs,Old_Model_Training.LX_Pred,algorithm_names(3), results_training);
validation_results = struct();
result_trained_model.old_model_results = old_model_results;
result_trained_model.old_model_results.validation_results = validation_results;
result_trained_model.old_model_results.validation_results.validation_predictions = Old_Model_Training.LX_Pred;
result_trained_model.old_model_results.validation_results.metrics = results_training("old_model",:);

% compute test performance
results_test = compute_metrics(Old_Model_Test.LX_Obs,Old_Model_Test.LX_Pred,algorithm_names(3), results_test);
test_results = struct();
result_trained_model.old_model_results.test_results = test_results;
result_trained_model.old_model_results.test_results.test_predictions = Old_Model_Test.LX_Pred;
result_trained_model.old_model_results.test_results.metrics = results_test("old_model",:);

clc;
close all;

writetable(results_training, '1-Trained-Models/Trained-Test-Results-k-4-old-model-configuration/Results-calibration-model-k-4.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/Trained-Test-Results-k-4-old-model-configuration/Results-test-model-k-4.xlsx', 'WriteRowNames',true);
save("1-Trained-Models/Trained-Test-Results-k-4-old-model-configuration/Trained-Tested-model-k-4.mat","result_trained_model");