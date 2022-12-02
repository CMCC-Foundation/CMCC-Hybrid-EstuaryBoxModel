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
filepath = "0-Dataset\CK_OBS_WITH_FEATURES.xlsx";
nVars = 7;
dataRange = "A2:G1462";
sheetName = "Ck_Old_Model";
varNames = ["Year","Qriver", "Qtide", "Socean", "CkObs", "CkOldModel"]; 
varTypes = ["int16", "double", "double", "double", "double","double"];

[ck_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Remove observations with missing values
ck_dataset = remove_missing_data_features(ck_dataset);

%% Split original dataset in training and test set
[ck_training_dataset, ck_test_dataset] = create_training_test_dataset(ck_dataset, 0.2);

save('0-Dataset/training_test_2016_2019/Ck-Training-Dataset.mat','ck_training_dataset');
save('0-Dataset/training_test_2016_2019/Ck-Test-Dataset.mat','ck_test_dataset');
writetable(ck_training_dataset, '0-Dataset/training_test_2016_2019/Ck-Training-Dataset.xlsx', 'WriteRowNames',true);
writetable(ck_test_dataset, '0-Dataset/training_test_2016_2019/Ck-Test-Dataset.xlsx', 'WriteRowNames',true);

%% Plot boxplot for training and test dataset
plot_boxplot_training_test("Boxplot of features for ck estimation",...
     removevars(ck_training_dataset,{'Year','CkOldModel'}),...
     removevars(ck_test_dataset,{'Year','CkOldModel'}));

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'neural_network' };

results_training = table('Size', [3 8], ...
    'VariableTypes', {'double','double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

%% Set target feature for the machine and deep learning model
targetFeatureName = 'CkObs';

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
result_trained_model.random_forest = random_forest_function(removevars(ck_training_dataset, {'Year', 'CkOldModel'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);

% save test results
test_results = struct();
result_trained_model.random_forest.test_results = test_results;
result_trained_model.random_forest.test_results.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(ck_test_dataset, {'Year', 'CkOldModel'}));
results_test= compute_metrics(ck_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for ck estimation with Random forest");


%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.lsboost = lsboost_function(removevars(ck_training_dataset, {'Year', 'CkOldModel'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for ck estimation with Lsboost");

% save test results
test_results = struct();
result_trained_model.lsboost.test_results = test_results;
result_trained_model.lsboost.test_results.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(ck_test_dataset, {'Year','CkOldModel'}));
results_test= compute_metrics(ck_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);


%% Training neural network model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.neural_network = neural_network_function(removevars(ck_training_dataset, {'Year', 'CkOldModel'}),targetFeatureName,1,3, 10, 100,max_objective_evaluations, k);
results_training = compute_metrics(ck_training_dataset(:, targetFeatureName), result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(3), results_training);
result_trained_model.neural_network.validation_results.metrics = results_training("neural_network",:);

% save test result
test_results = struct();
result_trained_model.neural_network.test_results = test_results;
result_trained_model.neural_network.test_results.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(ck_test_dataset, {'Year', 'CkOldModel'}));
results_test= compute_metrics(ck_test_dataset(:, targetFeatureName), result_trained_model.neural_network.test_results.test_predictions, algorithm_names(3), results_test);
result_trained_model.neural_network.test_results.metrics = results_test("neural_network",:);

writetable(results_training, '1-Trained-Models/training_test_2016_2019/Results-ck-training-model.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/training_test_2016_2019/Results-ck-test-model.xlsx', 'WriteRowNames',true);
save("1-Trained-Models\training_test_2016_2019\Ck-Trained-Tested-model.mat","result_trained_model");