%% The following script group all the experiment carried out in this paper:
%  Given the dataset, we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 10. 
%  We use data from 2016-2017 to train and validate our model, and data 
%  from 2018-2019 examples to test it.
%  The aim is to compare the old model performance with these new models.

%% Add to path subdirectory
addpath(genpath('0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model'));

%% Set import dataset settings
filepath = "0-Dataset\CK_OBS_WITH_FEATURES.xlsx";
nVars = 6;
dataRange = "A2:F1462";
sheetName = "Ck_Old_Model";
varNames = ["Year","Q_river", "Q_tide", "S_ocean", "CK_Obs", "Ck_old_model"]; 
varTypes = ["int16", "double", "double", "double", "double","double"];

[ck_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Remove observations with missing values
ck_dataset = remove_missing_data_features(ck_dataset);

%% Split original dataset in training and test set
ck_training_dataset = ck_dataset(ck_dataset.Year == 2016 | ck_dataset.Year == 2017, :);
ck_test_dataset_2018 = ck_dataset(ck_dataset.Year == 2018, :);
ck_test_dataset_2019 = ck_dataset(ck_dataset.Year == 2019, :);
ck_test_dataset_2018_2019 = ck_dataset(ck_dataset.Year == 2018 | ck_dataset.Year == 2019, :);

save('0-Dataset/training_2016_2017_test_2018_2019_comparing_old_model/CK_OLD_MODEL_PREDICTIONS.mat', 'ck_dataset');
save('0-Dataset/training_2016_2017_test_2018_2019_comparing_old_model/Ck-Training-Dataset_2016_2017.mat','ck_training_dataset');
save('0-Dataset/training_2016_2017_test_2018_2019_comparing_old_model/Ck-Test-Dataset_2018_2019.mat','ck_test_dataset_2018_2019');

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'neural_network', 'old_model' };

results_training = table('Size', [4 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_2018_dataset = table('Size', [4 8], ...
    'VariableTypes', {'double', 'double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_2019_dataset = table('Size', [4 8], ...
    'VariableTypes', {'double', 'double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_2018_2019_dataset = table('Size', [4 8], ...
    'VariableTypes', {'double', 'double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

%% Set target feature for the machine and deep learning model
targetFeatureName = 'CK_Obs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 60;

%% Set k to be use in k-fold cross validation
k = 10;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(ck_training_dataset, {'Year', 'Ck_old_model'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);

% save test results
test_2018_dataset = struct();
test_2019_dataset = struct();
test_2018_2019_dataset = struct();
result_trained_model.random_forest.test_results.test_2018_dataset = test_2018_dataset;
result_trained_model.random_forest.test_results.test_2019_dataset = test_2019_dataset;
result_trained_model.random_forest.test_results.test_2018_2019_dataset = test_2018_2019_dataset;

% test only on 2018 observations
result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(ck_test_dataset_2018, {'Year','CK_Obs','Ck_old_model'}));
results_test_2018_dataset = compute_metrics(ck_test_dataset_2018(:, targetFeatureName), result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions, algorithm_names(1), results_test_2018_dataset);
result_trained_model.random_forest.test_results.test_2018_dataset.metrics = results_test_2018_dataset("random_forest",:);

% test only on 2019 observations 
result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(ck_test_dataset_2019, {'Year','CK_Obs','Ck_old_model'}));
results_test_2019_dataset = compute_metrics(ck_test_dataset_2019(:, targetFeatureName), result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions, algorithm_names(1), results_test_2019_dataset);
result_trained_model.random_forest.test_results.test_2019_dataset.metrics = results_test_2019_dataset("random_forest",:);

% test on 2018-2019 observations
result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(ck_test_dataset_2018_2019, {'Year','CK_Obs','Ck_old_model'}));
results_test_2018_2019_dataset = compute_metrics(ck_test_dataset_2018_2019(:, targetFeatureName), result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(1), results_test_2018_2019_dataset);
result_trained_model.random_forest.test_results.test_2018_2019_dataset.metrics = results_test_2018_2019_dataset("random_forest",:);


%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.lsboost = lsboost_function(removevars(ck_training_dataset, {'Year', 'Ck_old_model'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);

% save test results
test_2018_dataset = struct();
test_2019_dataset = struct();
test_2018_2019_dataset = struct();
result_trained_model.lsboost.test_results.test_2018_dataset = test_2018_dataset;
result_trained_model.lsboost.test_results.test_2019_dataset = test_2019_dataset;
result_trained_model.lsboost.test_results.test_2018_2019_dataset = test_2018_2019_dataset;

% test only on 2018 observations
result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(ck_test_dataset_2018,  {'Year','CK_Obs','Ck_old_model'}));
results_test_2018_dataset = compute_metrics(ck_test_dataset_2018(:, targetFeatureName), result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions, algorithm_names(2), results_test_2018_dataset);
result_trained_model.lsboost.test_results.test_2018_dataset.metrics = results_test_2018_dataset("lsboost",:);

% test only on 2019 observations 
result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(ck_test_dataset_2019,  {'Year','CK_Obs','Ck_old_model'}));
results_test_2019_dataset = compute_metrics(ck_test_dataset_2019(:, targetFeatureName), result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions, algorithm_names(2), results_test_2019_dataset);
result_trained_model.lsboost.test_results.test_2019_dataset.metrics = results_test_2019_dataset("lsboost",:);

% test on 2018-2019 observations
result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(ck_test_dataset_2018_2019,  {'Year','CK_Obs','Ck_old_model'}));
results_test_2018_2019_dataset = compute_metrics(ck_test_dataset_2018_2019(:, targetFeatureName), result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(2), results_test_2018_2019_dataset);
result_trained_model.lsboost.test_results.test_2018_2019_dataset.metrics = results_test_2018_2019_dataset("lsboost",:);


%% Training neural network model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.neural_network = neural_network_function(removevars(ck_training_dataset, {'Year', 'Ck_old_model'}),targetFeatureName,1,3, 10, 50,max_objective_evaluations, k);
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(3), results_training);
result_trained_model.neural_network.validation_results.metrics = results_training("neural_network",:);

% save test results
test_2018_dataset = struct();
test_2019_dataset = struct();
test_2018_2019_dataset = struct();
result_trained_model.neural_network.test_results.test_2018_dataset = test_2018_dataset;
result_trained_model.neural_network.test_results.test_2019_dataset = test_2019_dataset;
result_trained_model.neural_network.test_results.test_2018_2019_dataset = test_2018_2019_dataset;

% test only on 2018 observations
result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(ck_test_dataset_2018,  {'Year','CK_Obs','Ck_old_model'}));
results_test_2018_dataset = compute_metrics(ck_test_dataset_2018(:, targetFeatureName), result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions, algorithm_names(3), results_test_2018_dataset);
result_trained_model.neural_network.test_results.test_2018_dataset.metrics = results_test_2018_dataset("neural_network",:);

% test only on 2019 observations 
result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(ck_test_dataset_2019, {'Year','CK_Obs','Ck_old_model'}));
results_test_2019_dataset = compute_metrics(ck_test_dataset_2019(:, targetFeatureName), result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions, algorithm_names(3), results_test_2019_dataset);
result_trained_model.neural_network.test_results.test_2019_dataset.metrics = results_test_2019_dataset("neural_network",:);

% test on 2018-2019 observations
result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(ck_test_dataset_2018_2019,  {'Year','CK_Obs','Ck_old_model'}));
results_test_2018_2019_dataset = compute_metrics(ck_test_dataset_2018_2019(:, targetFeatureName), result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(3), results_test_2018_2019_dataset);
result_trained_model.neural_network.test_results.test_2018_2019_dataset.metrics = results_test_2018_2019_dataset("neural_network",:);

%% Update metrics from old model
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), ck_training_dataset(:,"Ck_old_model"), algorithm_names(4), results_training);
results_test_2018_dataset = compute_metrics(ck_test_dataset_2018(:, targetFeatureName), ck_test_dataset_2018(:,"Ck_old_model"), algorithm_names(4), results_test_2018_dataset);
results_test_2019_dataset = compute_metrics(ck_test_dataset_2019(:, targetFeatureName), ck_test_dataset_2019(:,"Ck_old_model"), algorithm_names(4), results_test_2019_dataset);
results_test_2018_2019_dataset = compute_metrics(ck_test_dataset_2018_2019(:, targetFeatureName), ck_test_dataset_2018_2019(:,"Ck_old_model"), algorithm_names(4), results_test_2018_2019_dataset);

%% close plot optimization 
clc;
close all;

%% Save results
writetable(results_training, '1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Results-ck-calibration-model-k-10-old-configuration.xlsx', 'WriteRowNames',true);
writetable(results_test_2018_dataset, '1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Results-ck-test-2018-model-k-10-old-configuration.xlsx', 'WriteRowNames',true);
writetable(results_test_2019_dataset, '1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Results-ck-test-2019-model-k-10-old-configuration.xlsx', 'WriteRowNames',true);
writetable(results_test_2018_2019_dataset, '1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Results-ck-test-2018-2019-model-k-10-old-configuration.xlsx', 'WriteRowNames',true);
save("1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Ck-Trained-Tested-model-k-10-old-configuration.mat","result_trained_model");