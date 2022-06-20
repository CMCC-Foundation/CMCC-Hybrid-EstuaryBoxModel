%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 4. 
%  In the end, some figures are plotted and dataset and model are saved. 

%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('1-Pre-Processing'));
addpath(genpath('2-Machine-Learning-Function'));
addpath(genpath('3_Trained-Models'));

filepath = "0-Dataset\LX_OBS_WITH_FEATURES.xlsx";
[lx_dataset] = import_dataset(filepath);
save('0-Dataset/LX_OBS_WITH_FEATURES.mat', ...
    'lx_dataset');

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Lx_OBS';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 3;

%% Removed useless features
training_dataset = lx_dataset;
training_dataset = removevars(training_dataset, {'DATE','Dataset_Type'});

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'neural_network' };

results = table('Size', [3 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

k = 4;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.random_forest = random_forest_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
result_trained_model.random_forest.metrics.mae = computeMAE(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.rse = computeRSE(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.rrse = sqrt(result_trained_model.random_forest.metrics.rse);
result_trained_model.random_forest.metrics.rae = computeRAE(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.r2 = computeR2(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
corr_coeff_matrix = corrcoef(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions);
result_trained_model.random_forest.metrics.corr_coeff = corr_coeff_matrix(1,2);

results("random_forest","RMSE") = {result_trained_model.random_forest.metrics.rmse};
results("random_forest","MAE") = {result_trained_model.random_forest.metrics.mae};
results("random_forest","RSE") = {result_trained_model.random_forest.metrics.rse};
results("random_forest","RRSE") = {result_trained_model.random_forest.metrics.rrse};
results("random_forest","RAE") = {result_trained_model.random_forest.metrics.rae};
results("random_forest","R2") = {result_trained_model.random_forest.metrics.r2};
results("random_forest","Corr Coeff") = {result_trained_model.random_forest.metrics.corr_coeff};

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
corr_coeff_matrix = corrcoef(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions);
result_trained_model.lsboost.metrics.corr_coeff = corr_coeff_matrix(1,2);

results("lsboost","RMSE") = {result_trained_model.lsboost.metrics.rmse};
results("lsboost","MAE") = {result_trained_model.lsboost.metrics.mae};
results("lsboost","RSE") = {result_trained_model.lsboost.metrics.rse};
results("lsboost","RRSE") = {result_trained_model.lsboost.metrics.rrse};
results("lsboost","RAE") = {result_trained_model.lsboost.metrics.rae};
results("lsboost","R2") = {result_trained_model.lsboost.metrics.r2};
results("lsboost","Corr Coeff") = {result_trained_model.lsboost.metrics.corr_coeff};


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

results("neural_network","RMSE") = {result_trained_model.neural_network.metrics.rmse};
results("neural_network","MAE") = {result_trained_model.neural_network.metrics.mae};
results("neural_network","RSE") = {result_trained_model.neural_network.metrics.rse};
results("neural_network","RRSE") = {result_trained_model.neural_network.metrics.rrse};
results("neural_network","RAE") = {result_trained_model.neural_network.metrics.rae};
results("neural_network","R2") = {result_trained_model.neural_network.metrics.r2};
results("neural_network","Corr Coeff") = {result_trained_model.neural_network.metrics.corr_coeff};

clc;
close all;

writetable(results, '3-Trained-Models/Results-Trained-model-k-4.xlsx', 'WriteRowNames',true);
save("3-Trained-Models\Trained-model-k-4.mat","result_trained_model");


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