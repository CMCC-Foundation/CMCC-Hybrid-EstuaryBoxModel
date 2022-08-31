%% The following script group all the experiment carried out in this paper:
%  Given the dataset, we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 10. 
%  We use 70% of examples to train and validate our model, and 30% examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset\training_test_2016_2019'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('1_Trained-Models\training_test_2016_2019'));

%% Set import dataset settings
filepath = "0-Dataset\training_test_2016_2019\SALINITY_OBS_WITH_FEATURES.xlsx";
nVars = 7;
dataRange = "A2:G1462";
sheetName = "Salinity_obs";
varNames = ["Year","Q_river", "Q_ll", "Q_tide", "S_ll", "S_ocean", "Salinity_Obs"]; 
varTypes = ["int16", "double", "double", "double", "double","double","double"];

[salinity_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Salinity_Obs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 30;

%% Remove observations with missing values
salinity_dataset = remove_missing_data_features(salinity_dataset);

%% Split original dataset in training and test set
[salinity_training_dataset, salinity_test_dataset] = create_training_test_dataset(salinity_dataset, 0.3);

save('0-Dataset/training_test_2016_2019/SALINITY_OBS_WITH_FEATURES.mat', ...
    'salinity_dataset');
save('0-Dataset/training_test_2016_2019/Salinity-Training-Dataset.mat', ...
    'salinity_training_dataset');
save('0-Dataset/training_test_2016_2019/Salinity-Test-Dataset.mat', ...
    'salinity_test_dataset');

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

k = 10;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(salinity_training_dataset, {'Year'}),targetFeatureName,max_objective_evaluations, k);
result_trained_model.random_forest.metrics.mae = computeMAE(salinity_training_dataset.Salinity_Obs, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.rse = computeRSE(salinity_training_dataset.Salinity_Obs, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.rrse = sqrt(result_trained_model.random_forest.metrics.rse);
result_trained_model.random_forest.metrics.rae = computeRAE(salinity_training_dataset.Salinity_Obs, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.r2 = computeR2(salinity_training_dataset.Salinity_Obs, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.corr_coeff = computeCorrCoef(salinity_training_dataset.Salinity_Obs, result_trained_model.random_forest.predictions);

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
result_trained_model.random_forest.test.predictions = result_trained_model.random_forest.model.predictFcn(removevars(salinity_test_dataset, {'Year'}));
result_trained_model.random_forest.test.metrics = metrics;
result_trained_model.random_forest.test.metrics.rmse = computeRMSE(salinity_test_dataset.Salinity_Obs, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.mae = computeMAE(salinity_test_dataset.Salinity_Obs, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.rse = computeRSE(salinity_test_dataset.Salinity_Obs, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.rrse = sqrt(result_trained_model.random_forest.test.metrics.rse);
result_trained_model.random_forest.test.metrics.rae = computeRAE(salinity_test_dataset.Salinity_Obs, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.r2 = computeR2(salinity_test_dataset.Salinity_Obs, result_trained_model.random_forest.test.predictions);
result_trained_model.random_forest.test.metrics.corr_coeff = computeCorrCoef(salinity_test_dataset.Salinity_Obs, result_trained_model.random_forest.test.predictions);

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

result_trained_model.lsboost = lsboost_function(removevars(salinity_training_dataset, {'Year'}),targetFeatureName,max_objective_evaluations, k);
result_trained_model.lsboost.metrics.mae = computeMAE(salinity_training_dataset.Salinity_Obs, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.rse = computeRSE(salinity_training_dataset.Salinity_Obs, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.rrse = sqrt(result_trained_model.lsboost.metrics.rse);
result_trained_model.lsboost.metrics.rae = computeRAE(salinity_training_dataset.Salinity_Obs, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.r2 = computeR2(salinity_training_dataset.Salinity_Obs, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.corr_coeff = computeCorrCoef(salinity_training_dataset.Salinity_Obs, result_trained_model.lsboost.predictions);

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
result_trained_model.lsboost.test.predictions = result_trained_model.lsboost.model.predictFcn(removevars(salinity_test_dataset, {'Year'}));
result_trained_model.lsboost.test.metrics = metrics;
result_trained_model.lsboost.test.metrics.rmse = computeRMSE(salinity_test_dataset.Salinity_Obs, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.mae = computeMAE(salinity_test_dataset.Salinity_Obs, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.rse = computeRSE(salinity_test_dataset.Salinity_Obs, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.rrse = sqrt(result_trained_model.lsboost.test.metrics.rse);
result_trained_model.lsboost.test.metrics.rae = computeRAE(salinity_test_dataset.Salinity_Obs, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.r2 = computeR2(salinity_test_dataset.Salinity_Obs, result_trained_model.lsboost.test.predictions);
result_trained_model.lsboost.test.metrics.corr_coeff = computeCorrCoef(salinity_test_dataset.Salinity_Obs, result_trained_model.lsboost.test.predictions);

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

result_trained_model.neural_network = neural_network_function(removevars(salinity_training_dataset, {'Year'}),targetFeatureName,1,3, 10, 50,max_objective_evaluations, k);
result_trained_model.neural_network.metrics.mae = computeMAE(salinity_training_dataset.Salinity_Obs, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.rse = computeRSE(salinity_training_dataset.Salinity_Obs, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.rrse = sqrt(result_trained_model.neural_network.metrics.rse);
result_trained_model.neural_network.metrics.rae = computeRAE(salinity_training_dataset.Salinity_Obs, result_trained_model.neural_network.predictions);
result_trained_model.neural_network.metrics.r2 = computeR2(salinity_training_dataset.Salinity_Obs, result_trained_model.neural_network.predictions);
corr_coeff_matrix = corrcoef(salinity_training_dataset.Salinity_Obs, result_trained_model.neural_network.predictions);
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
result_trained_model.neural_network.test.predictions = result_trained_model.neural_network.model.predictFcn(removevars(salinity_test_dataset, {'Year'}));
result_trained_model.neural_network.test.metrics = metrics;
result_trained_model.neural_network.test.metrics.rmse = computeRMSE(salinity_test_dataset.Salinity_Obs, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.mae = computeMAE(salinity_test_dataset.Salinity_Obs, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.rse = computeRSE(salinity_test_dataset.Salinity_Obs, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.rrse = sqrt(result_trained_model.neural_network.test.metrics.rse);
result_trained_model.neural_network.test.metrics.rae = computeRAE(salinity_test_dataset.Salinity_Obs, result_trained_model.neural_network.test.predictions);
result_trained_model.neural_network.test.metrics.r2 = computeR2(salinity_test_dataset.Salinity_Obs, result_trained_model.neural_network.test.predictions);
corr_coeff_matrix = corrcoef(salinity_test_dataset.Salinity_Obs, result_trained_model.neural_network.test.predictions);
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

writetable(results_training, '1-Trained-Models/training_test_2016_2019/Results-salinity-calibration-model-k-10.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/training_test_2016_2019/Results-salinity-test-model-k-10.xlsx', 'WriteRowNames',true);
save("1-Trained-Models\training_test_2016_2019\Salinity-Trained-Tested-model-k-10.mat","result_trained_model");


function [rmse] = computeRMSE(obs, pred)
    rmse = sqrt(sum((obs - pred).^2)/height(obs));
end

function [nrmse] = computeNRMSE(obs, pred)
    nrmse = computeRMSE(obs, pred) / mean(obs);
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