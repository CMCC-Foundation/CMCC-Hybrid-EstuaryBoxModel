%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 5. 
%  We use 20 examples to train and validate our model, and 5 examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1-Trained-Models'));

%% Set import dataset settings
filepath = "0-Dataset\LX_OBS_WITH_FEATURES.xlsx";
nVars = 8;
dataRange = "A2:H26";
sheetName = "Lx_obs";
varNames = ["DATE","Q_l", "Q_r", "S_l", "Q_tide", "Lx_OBS", "Lx_Model", "Dataset_Type"]; 
varTypes = ["datetime", "double", "double", "double", "double","double", "double","categorical"];

[lx_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);
save('0-Dataset/LX_OBS_WITH_FEATURES.mat','lx_dataset');

%% Scatter plot
plot_scatterplot(lx_dataset(:,2:5), lx_dataset(:,6));

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Lx_OBS';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 45;

%% Removed useless features and split dataset in training and test set
lx_training_dataset = lx_dataset(strcmp(string(lx_dataset.Dataset_Type), "CALIBRATION RANGE"),:);
lx_test_dataset = lx_dataset(strcmp(string(lx_dataset.Dataset_Type), 'VALIDATION RANGE'),:);

%% Plot boxplot for training and test dataset
plot_boxplot_training_test("Boxplot of features for lx estimation",...
     removevars(lx_training_dataset,{'DATE','Lx_Model','Dataset_Type'}),...
     removevars(lx_test_dataset,{'DATE','Lx_Model','Dataset_Type'}));

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'old_model'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

results_training = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE','MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

pwbTable = table('Size',[numel(pwbX) numel(algorithm_names)],...
    'VariableTypes', repmat({'double'}, 1, numel(algorithm_names)), ...
    'VariableNames', algorithm_names,...
    'RowNames', pwbXRowNames);

result_trained_model = struct();

k = 5;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(lx_training_dataset, {'DATE','Dataset_Type', 'Lx_Model'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for Lx estimation with Random Forest");

% save test results
result_trained_model.random_forest.test_results.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(lx_test_dataset, {'DATE','Dataset_Type', 'Lx_Model'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.test_predictions,pwbTable,algorithm_names(1),pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.lsboost = lsboost_function(removevars(lx_training_dataset, {'DATE','Dataset_Type', 'Lx_Model'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for Lx estimation with Lsboost");

% save test results
result_trained_model.lsboost.test_results.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_dataset, {'DATE','Dataset_Type', 'Lx_Model'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.test_predictions,pwbTable,algorithm_names(2),pwbX);

%% Compute metrics on old model
old_model_results = struct();

% compute training performance
results_training = compute_metrics(lx_training_dataset.Lx_OBS,lx_training_dataset.Lx_Model,algorithm_names(3), results_training);
validation_results = struct();
result_trained_model.old_model_results = old_model_results;
result_trained_model.old_model_results.validation_results = validation_results;
result_trained_model.old_model_results.validation_results.validation_predictions = lx_training_dataset.Lx_Model;
result_trained_model.old_model_results.validation_results.metrics = results_training("old_model",:);

% compute test performance
results_test = compute_metrics(lx_training_dataset.Lx_OBS,lx_training_dataset.Lx_Model,algorithm_names(3), results_test);
test_results = struct();
result_trained_model.old_model_results.test_results = test_results;
result_trained_model.old_model_results.test_results.test_predictions = lx_test_dataset.Lx_Model;
result_trained_model.old_model_results.test_results.metrics = results_test("old_model",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.old_model_results.test_results.test_predictions,pwbTable,algorithm_names(3),pwbX);

clc;
disp(results_training);
disp(results_test);
disp(pwbTable);

writetable(results_training, '1-Trained-Models/Trained-Test-Results-k-5-old-model-configuration/Results-calibration-model-k-5.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/Trained-Test-Results-k-5-old-model-configuration/Results-test-model-k-5.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "1-Trained-Models/Trained-Test-Results-k-5-old-model-configuration/pwbTable.xlsx", "WriteRowNames", true);
save("1-Trained-Models/Trained-Test-Results-k-5-old-model-configuration/Trained-Tested-model-k-5.mat","result_trained_model");