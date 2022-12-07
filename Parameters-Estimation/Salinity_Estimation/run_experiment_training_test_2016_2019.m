%% The following script group all the experiment carried out in this paper:
%  Given the dataset, we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 5. 
%  We use 80% of examples to train and validate our model, and 20% examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset\training_test_2016_2019'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1_Trained-Models\training_test_2016_2019'));

%% Set import dataset settings
filepath = "0-Dataset\training_test_2016_2019\SALINITY_OBS_WITH_FEATURES.xlsx";
nVars = 7;
dataRange = "A2:G1462";
sheetName = "Salinity_obs";
varNames = ["Year","Qriver", "Qll", "Qtide", "Sll", "Socean", "SalinityObs"]; 
varTypes = ["int16","double", "double", "double", "double","double","double"];

[salinity_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Remove observations with missing values
salinity_dataset = remove_missing_data_features(salinity_dataset);

%% Split original dataset in training and test set
[salinity_training_dataset, salinity_test_dataset] = create_training_test_dataset(salinity_dataset, 0.2);

save('0-Dataset/training_test_2016_2019/Salinity-Training-Dataset.mat','salinity_training_dataset');
save('0-Dataset/training_test_2016_2019/Salinity-Test-Dataset.mat','salinity_test_dataset');
writetable(salinity_training_dataset, '0-Dataset/training_test_2016_2019/Salinity-Training-Dataset.xlsx', 'WriteRowNames',true);
writetable(salinity_test_dataset, '0-Dataset/training_test_2016_2019/Salinity-Test-Dataset.xlsx', 'WriteRowNames',true);

%% Plot boxplot for training and test dataset
plot_boxplot_training_test("Boxplot of features for salinity estimation",...
     removevars(salinity_training_dataset,{'Year'}),...
     removevars(salinity_test_dataset,{'Year'}));

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'neural_network'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

results_training = table('Size', [3 8], ...
    'VariableTypes', {'double','double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

pwbTable = table('Size',[numel(pwbX) numel(algorithm_names)],...
    'VariableTypes', repmat({'double'}, 1, numel(algorithm_names)), ...
    'VariableNames', algorithm_names,...
    'RowNames', pwbXRowNames);

result_trained_model = struct();

%% Set target feature for the machine and deep learning model
targetFeatureName = 'SalinityObs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 45;

%% Set k to be use in k-fold cross validation
k = 5;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(salinity_training_dataset, {'Year'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for salinity estimation with Random Forest");

% save test results
test_results = struct();
result_trained_model.random_forest.test_results = test_results;
result_trained_model.random_forest.test_results.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(salinity_test_dataset, {'Year'}));
results_test= compute_metrics(salinity_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);
pwbTable = create_pwb_table(salinity_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.test_predictions,pwbTable,algorithm_names(1),pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.lsboost = lsboost_function(removevars(salinity_training_dataset, {'Year'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for salinity estimation with Lsboost");

% save test results
test_results = struct();
result_trained_model.lsboost.test_results = test_results;
result_trained_model.lsboost.test_results.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(salinity_test_dataset, {'Year'}));
results_test= compute_metrics(salinity_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);
pwbTable = create_pwb_table(salinity_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.test_predictions,pwbTable,algorithm_names(2),pwbX);

%% Training neural network model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.neural_network = neural_network_function(removevars(salinity_training_dataset, {'Year'}),targetFeatureName,1,3, 10, 100,max_objective_evaluations, k);
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(3), results_training);
result_trained_model.neural_network.validation_results.metrics = results_training("neural_network",:);

% save test result
test_results = struct();
result_trained_model.neural_network.test_results = test_results;
result_trained_model.neural_network.test_results.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(salinity_test_dataset, {'Year'}));
results_test= compute_metrics(salinity_test_dataset(:, targetFeatureName), result_trained_model.neural_network.test_results.test_predictions, algorithm_names(3), results_test);
result_trained_model.neural_network.test_results.metrics = results_test("neural_network",:);
pwbTable = create_pwb_table(salinity_test_dataset(:, targetFeatureName), result_trained_model.neural_network.test_results.test_predictions,pwbTable,algorithm_names(3),pwbX);

clc;
disp(results_training);
disp(results_test);
disp(pwbTable);

writetable(results_training, '1-Trained-Models/training_test_2016_2019/Results-salinity-training-model.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/training_test_2016_2019/Results-salinity-test-model.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "1-Trained-Models/training_test_2016_2019/pwbTable.xlsx", "WriteRowNames", true);
save("1-Trained-Models\training_test_2016_2019\Salinity-Trained-Tested-model.mat","result_trained_model");