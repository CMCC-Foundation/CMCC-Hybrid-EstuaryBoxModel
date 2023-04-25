% The following script group all the experiment carried out in this paper:
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
% import training dataset
filepath = "0-Dataset\lx-train-dataset.xlsx";
nVars = 7;
dataRange = "A2:G21";
sheetName = "Lx_obs";
varNames = ["Date","Qll", "Qriver", "Sll", "Qtide", "LxObs", "LxObsOldModel"]; 
varTypes = ["datetime", "double", "double", "double", "double","double", "double"];

lx_training_dataset = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);
save('0-Dataset/lx_training_dataset.mat','lx_training_dataset');

% import test dataset
filepath = "0-Dataset\lx-test-dataset.xlsx";
nVars = 7;
dataRange = "A2:G6";
sheetName = "Lx_obs";
varNames = ["Date","Qll", "Qriver", "Sll", "Qtide", "LxObs", "LxObsOldModel"]; 
varTypes = ["datetime", "double", "double", "double", "double","double", "double"];

lx_test_dataset = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);
save('0-Dataset/lx_test_dataset.mat','lx_test_dataset');

%% Set target feature for the machine and deep learning model
targetFeatureName = 'LxObs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 60;

%% Create table for k-fold cross validation results
algorithm_names = {'EBM','RF','LSBoost'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

pwbTable = table('Size',[numel(pwbX) numel(algorithm_names)],...
    'VariableTypes', repmat({'double'}, 1, numel(algorithm_names)), ...
    'VariableNames', algorithm_names,...
    'RowNames', pwbXRowNames);

results_training = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [3 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE','MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

k = 5;

%% Compute metrics on EBM
EBM = struct();

% compute training performance
results_training = compute_metrics(lx_training_dataset.LxObs,lx_training_dataset.LxObsOldModel,algorithm_names(1), results_training);
validation_results = struct();
result_trained_model.EBM = EBM;
result_trained_model.EBM.validation_results = validation_results;
result_trained_model.EBM.validation_results.validation_predictions = lx_training_dataset.LxObsOldModel;
result_trained_model.EBM.validation_results.metrics = results_training("EBM",:);

% compute test performance
results_test = compute_metrics(lx_test_dataset.LxObs,lx_test_dataset.LxObsOldModel,algorithm_names(1), results_test);
test_results = struct();
result_trained_model.EBM.test_results = test_results;
result_trained_model.EBM.test_results.test_predictions = lx_test_dataset.LxObsOldModel;
result_trained_model.EBM.test_results.metrics = results_test("EBM",:);
pwbTable = create_pwb_table( ...
    lx_test_dataset.LxObs, ...
    result_trained_model.EBM.test_results.test_predictions, ...
    pwbTable, ...
    algorithm_names(1), ...
    pwbX);

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.RF = random_forest_function(...
    lx_training_dataset(:,["Qll","Qriver","Qtide","Sll","LxObs"]), ...
    targetFeatureName, ...
    max_objective_evaluations, ...
    k);

results_training = compute_metrics(lx_training_dataset.LxObs,...
    result_trained_model.RF.validation_results.validation_predictions, ...
    algorithm_names(2), ...
    results_training);

result_trained_model.RF.validation_results.metrics = results_training("RF",:);
plot_importance(result_trained_model.RF.feature_importance, ...
    "Features importance for Lx estimation with Random Forest");

% save test results
result_trained_model.RF.test_results.test_predictions = ...
    result_trained_model.RF.model.predictFcn(lx_test_dataset(:,["Qll","Qriver","Qtide","Sll"]));

results_test = compute_metrics( ...
    lx_test_dataset.LxObs, ...
    result_trained_model.RF.test_results.test_predictions, ...
    algorithm_names(2), ...
    results_test);

result_trained_model.RF.test_results.metrics = results_test("RF",:);
pwbTable = create_pwb_table(lx_test_dataset.LxObs, ...
    result_trained_model.RF.test_results.test_predictions, ...
    pwbTable,algorithm_names(2), ...
    pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.LSBoost = lsboost_function( ...
    lx_training_dataset(:,["Qll","Qriver","Qtide","Sll","LxObs"]), ...
    targetFeatureName, ...
    max_objective_evaluations, ...
    k);

results_training = compute_metrics( ...
    lx_training_dataset.LxObs, ...
    result_trained_model.LSBoost.validation_results.validation_predictions, ...
    algorithm_names(3), ...
    results_training);

result_trained_model.LSBoost.validation_results.metrics = results_training("LSBoost",:);
plot_importance(result_trained_model.LSBoost.feature_importance, "Features importance for Lx estimation with Lsboost");

% save test results
result_trained_model.LSBoost.test_results.test_predictions = ...
    result_trained_model.LSBoost.model.predictFcn(lx_test_dataset(:,["Qll","Qriver","Qtide","Sll"]));

results_test = compute_metrics( ...
    lx_test_dataset.LxObs, ...
    result_trained_model.LSBoost.test_results.test_predictions, ...
    algorithm_names(3), ...
    results_test);

result_trained_model.LSBoost.test_results.metrics = results_test("LSBoost",:);
pwbTable = create_pwb_table( ...
    lx_test_dataset.LxObs, ...
    result_trained_model.LSBoost.test_results.test_predictions, ...
    pwbTable, ...
    algorithm_names(3), ...
    pwbX);

%% display results
clc;
disp("Training results")
disp("--------------------------------------------------------")
disp(results_training(["RF","LSBoost"],["RMSE","Corr Coeff"]));
disp("Test results")
disp("--------------------------------------------------------")
disp(results_test(["RF","LSBoost"],["RMSE","Corr Coeff"]));
disp(pwbTable);

%% Save
writetable(results_training, '1-Trained-Models/Results-calibration-model-k-5.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models/Results-test-model-k-5.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "1-Trained-Models/pwbTable.xlsx", "WriteRowNames", true);
save("1-Trained-Models/Trained-Tested-model-k-5.mat","result_trained_model");