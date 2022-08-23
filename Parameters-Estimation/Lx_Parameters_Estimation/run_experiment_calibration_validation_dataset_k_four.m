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
addpath(genpath('1_Trained-Models'));

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
max_objective_evaluations = 30;

%% Removed useless features and split dataset in training and test set
training_dataset = lx_dataset(strcmp(string(lx_dataset.Dataset_Type), "CALIBRATION RANGE"),:);
training_dataset = removevars(training_dataset, {'DATE','Dataset_Type'});

testing_dataset = lx_dataset(strcmp(string(lx_dataset.Dataset_Type), 'VALIDATION RANGE'),:);
testing_dataset = removevars(testing_dataset, {'DATE','Dataset_Type'});

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'neural_network' };

results_training = table('Size', [3 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [3 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

k = 4;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
result_trained_model.random_forest.metrics.mae = computeMAE(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.rse = computeRSE(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.rrse = sqrt(result_trained_model.random_forest.metrics.rse);
result_trained_model.random_forest.metrics.rae = computeRAE(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.r2 = computeR2(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.corr_coeff = computeCorrCoef(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);

results_training("random_forest","RMSE") = {result_trained_model.random_forest.metrics.rmse};
results_training("random_forest","MAE") = {result_trained_model.random_forest.metrics.mae};
results_training("random_forest","RSE") = {result_trained_model.random_forest.metrics.rse};
results_training("random_forest","RRSE") = {result_trained_model.random_forest.metrics.rrse};
results_training("random_forest","RAE") = {result_trained_model.random_forest.metrics.rae};
results_training("random_forest","R2") = {result_trained_model.random_forest.metrics.r2};
results_training("random_forest","Corr Coeff") = {result_trained_model.random_forest.metrics.corr_coeff};

% save test results
test_performance = struct();
metrics = struct();
result_trained_model.random_forest.test = test_performance;
result_trained_model.random_forest.test.predictions = result_trained_model.random_forest.model.predictFcn(testing_dataset);
result_trained_model.random_forest.test.metrics = metrics;
result_trained_model.random_forest.test.metrics.rmse = computeRMSE(testing_dataset.Lx_OBS, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.mae = computeMAE(testing_dataset.Lx_OBS, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.rse = computeRSE(testing_dataset.Lx_OBS, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.rrse = sqrt(result_trained_model.random_forest.test.metrics.rse);
result_trained_model.random_forest.test.metrics.rae = computeRAE(testing_dataset.Lx_OBS, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.r2 = computeR2(testing_dataset.Lx_OBS, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.corr_coeff = computeCorrCoef(testing_dataset.Lx_OBS, result_trained_model.random_forest.test.predictions);

results_test("random_forest","RMSE") = {result_trained_model.random_forest.test.metrics.rmse};
results_test("random_forest","MAE") = {result_trained_model.random_forest.test.metrics.mae};
results_test("random_forest","RSE") = {result_trained_model.random_forest.test.metrics.rse};
results_test("random_forest","RRSE") = {result_trained_model.random_forest.test.metrics.rrse};
results_test("random_forest","RAE") = {result_trained_model.random_forest.test.metrics.rae};
results_test("random_forest","R2") = {result_trained_model.random_forest.test.metrics.r2};
results_test("random_forest","Corr Coeff") = {result_trained_model.random_forest.test.metrics.corr_coeff};


%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.lsboost = lsboost_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
result_trained_model.lsboost.metrics.mae = computeMAE(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.rse = computeRSE(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.rrse = sqrt(result_trained_model.lsboost.metrics.rse);
result_trained_model.lsboost.metrics.rae = computeRAE(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.r2 = computeR2(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.corr_coeff = computeCorrCoef(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions);

results_training("lsboost","RMSE") = {result_trained_model.lsboost.metrics.rmse};
results_training("lsboost","MAE") = {result_trained_model.lsboost.metrics.mae};
results_training("lsboost","RSE") = {result_trained_model.lsboost.metrics.rse};
results_training("lsboost","RRSE") = {result_trained_model.lsboost.metrics.rrse};
results_training("lsboost","RAE") = {result_trained_model.lsboost.metrics.rae};
results_training("lsboost","R2") = {result_trained_model.lsboost.metrics.r2};
results_training("lsboost","Corr Coeff") = {result_trained_model.lsboost.metrics.corr_coeff};

% save test results
test_performance = struct();
metrics = struct();
result_trained_model.lsboost.test = test_performance;
result_trained_model.lsboost.test.predictions = result_trained_model.lsboost.model.predictFcn(testing_dataset);
result_trained_model.lsboost.test.metrics = metrics;
result_trained_model.lsboost.test.metrics.rmse = computeRMSE(testing_dataset.Lx_OBS, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.mae = computeMAE(testing_dataset.Lx_OBS, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.rse = computeRSE(testing_dataset.Lx_OBS, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.rrse = sqrt(result_trained_model.lsboost.test.metrics.rse);
result_trained_model.lsboost.test.metrics.rae = computeRAE(testing_dataset.Lx_OBS, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.r2 = computeR2(testing_dataset.Lx_OBS, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.corr_coeff = computeCorrCoef(testing_dataset.Lx_OBS, result_trained_model.lsboost.test.predictions);

results_test("lsboost","RMSE") = {result_trained_model.lsboost.test.metrics.rmse};
results_test("lsboost","MAE") = {result_trained_model.lsboost.test.metrics.mae};
results_test("lsboost","RSE") = {result_trained_model.lsboost.test.metrics.rse};
results_test("lsboost","RRSE") = {result_trained_model.lsboost.test.metrics.rrse};
results_test("lsboost","RAE") = {result_trained_model.lsboost.test.metrics.rae};
results_test("lsboost","R2") = {result_trained_model.lsboost.test.metrics.r2};
results_test("lsboost","Corr Coeff") = {result_trained_model.lsboost.test.metrics.corr_coeff};


%% Training neural network model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.neural_network = neural_network_function(training_dataset,targetFeatureName,1,3, 10, 50,max_objective_evaluations, k);
result_trained_model.neural_network.metrics.mae = computeMAE(training_dataset.Lx_OBS, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.rse = computeRSE(training_dataset.Lx_OBS, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.rrse = sqrt(result_trained_model.neural_network.metrics.rse);
result_trained_model.neural_network.metrics.rae = computeRAE(training_dataset.Lx_OBS, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.r2 = computeR2(training_dataset.Lx_OBS, result_trained_model.neural_network.predictions);
corr_coeff_matrix = corrcoef(training_dataset.Lx_OBS, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.corr_coeff = corr_coeff_matrix(1,2);

results_training("neural_network","RMSE") = {result_trained_model.neural_network.metrics.rmse};
results_training("neural_network","MAE") = {result_trained_model.neural_network.metrics.mae};
results_training("neural_network","RSE") = {result_trained_model.neural_network.metrics.rse};
results_training("neural_network","RRSE") = {result_trained_model.neural_network.metrics.rrse};
results_training("neural_network","RAE") = {result_trained_model.neural_network.metrics.rae};
results_training("neural_network","R2") = {result_trained_model.neural_network.metrics.r2};
results_training("neural_network","Corr Coeff") = {result_trained_model.neural_network.metrics.corr_coeff};

% save test result
test_performance = struct();
metrics = struct();
result_trained_model.neural_network.test = test_performance;
result_trained_model.neural_network.test.predictions = result_trained_model.neural_network.model.predictFcn(testing_dataset);
result_trained_model.neural_network.test.metrics = metrics;
result_trained_model.neural_network.test.metrics.rmse = computeRMSE(testing_dataset.Lx_OBS, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.mae = computeMAE(testing_dataset.Lx_OBS, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.rse = computeRSE(testing_dataset.Lx_OBS, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.rrse = sqrt(result_trained_model.neural_network.test.metrics.rse);
result_trained_model.neural_network.test.metrics.rae = computeRAE(testing_dataset.Lx_OBS, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.r2 = computeR2(testing_dataset.Lx_OBS, result_trained_model.neural_network.test.predictions);
corr_coeff_matrix = corrcoef(testing_dataset.Lx_OBS, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.corr_coeff = corr_coeff_matrix(1,2);

results_test("neural_network","RMSE") = {result_trained_model.neural_network.test.metrics.rmse};
results_test("neural_network","MAE") = {result_trained_model.neural_network.test.metrics.mae};
results_test("neural_network","RSE") = {result_trained_model.neural_network.test.metrics.rse};
results_test("neural_network","RRSE") = {result_trained_model.neural_network.test.metrics.rrse};
results_test("neural_network","RAE") = {result_trained_model.neural_network.test.metrics.rae};
results_test("neural_network","R2") = {result_trained_model.neural_network.test.metrics.r2};
results_test("neural_network","Corr Coeff") = {result_trained_model.neural_network.test.metrics.corr_coeff};

clc;
close all;

writetable(results_training, '1-Trained-Models/Results-calibration-model-k-4.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/Results-test-model-k-4.xlsx', 'WriteRowNames',true);
save("1-Trained-Models\Trained-Tested-model-k-4.mat","result_trained_model");


function [rmse] = computeRMSE(obs, pred)
    rmse = sqrt(sum((obs - pred).^2)/height(obs));
end

function [mae] = computeMAE(obs, pred)
    mae = (sum(abs(pred-obs)))/height(obs);
end

function [rse] = computeRSE (obs, pred)
    num = sum((pred-obs).^2);
    den = sum((obs-mean(obs)).^2);
    rse = num/den;
end

function [rae] = computeRAE (obs, pred)
    num = sum(abs(pred-obs));
    den = sum(abs(mean(obs) - obs));
    rae = num / den;
end

function [r2] = computeR2 (obs, pred)
    sse = sum((obs-pred).^2);
    sst = sum((obs - mean(obs)).^2);
    r2 = 1 - (sse/sst);
end

function [r] = computeCorrCoef(obs, pred)
    corr_coeff_matrix = corrcoef(obs, pred);
    r = corr_coeff_matrix(1,2);
end