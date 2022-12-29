%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 5. 
%  We use 20 examples to train and validate our model, and 5 examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset\Po-all-branches\all-branches-merged'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1-Trained-Models\Po-all-branches\all-branches-merged'));

%% Set import dataset settings
filepath = "0-Dataset\Po-all-branches\all-branches-merged\LX_OBS_ALL_BRANCHES_MERGED.xlsx";
nVars = 9;
dataRange = "A2:I86";
sheetName = "ALL_BRANCHES";
varNames = ["DateObs","Qocean", "Qriver", "Qtide", "Sll", "LxObs", "LxOldEquationPred", "LxNewEquationPred","Branch"]; 
varTypes = ["datetime", "double", "double", "double", "double","double", "double", "double", "categorical"];

[lx_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Split original dataset in training and test set
[lx_training_dataset, lx_test_dataset] = create_training_test_dataset(lx_dataset, 0.2);

%% Set target feature for the machine and deep learning model
targetFeatureName = 'LxObs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 45;

%% Plot boxplot for training and test dataset
plot_boxplot_training_test("Boxplot of features for Lx estimation",...
     removevars(lx_training_dataset,{'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch'}),...
     removevars(lx_test_dataset,{'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch'}));

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'LxOldEquation', 'LxNewEquation'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

results_training = table('Size', [4 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [4 8], ...
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
result_trained_model.random_forest = random_forest_function(removevars(lx_training_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for Lx estimation with Random Forest");

% save test results
result_trained_model.random_forest.test_results.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(lx_test_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.test_predictions,pwbTable,algorithm_names(1),pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.lsboost = lsboost_function(removevars(lx_training_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for Lx estimation with Lsboost");

% save test results
result_trained_model.lsboost.test_results.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.test_predictions,pwbTable,algorithm_names(2),pwbX);

%% Compute metrics LxOldEquation
old_model_equation = struct();

% compute training performance
results_training = compute_metrics(lx_training_dataset.LxObs,lx_training_dataset.LxOldEquationPred,algorithm_names(3), results_training);
validation_results = struct();
result_trained_model.old_model_equation = old_model_equation;
result_trained_model.old_model_equation.validation_results = validation_results;
result_trained_model.old_model_equation.validation_results.validation_predictions = lx_training_dataset.LxOldEquationPred;
result_trained_model.old_model_equation.validation_results.metrics = results_training("LxOldEquation",:);

% compute test performance
results_test = compute_metrics(lx_test_dataset.LxObs,lx_test_dataset.LxOldEquationPred,algorithm_names(3), results_test);
test_results = struct();
result_trained_model.old_model_equation.test_results = test_results;
result_trained_model.old_model_equation.test_results.test_predictions = lx_test_dataset.LxOldEquationPred;
result_trained_model.old_model_equation.test_results.metrics = results_test("LxOldEquation",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.old_model_equation.test_results.test_predictions,pwbTable,algorithm_names(3),pwbX);

%% Compute metrics LxNewEquation
new_model_equation = struct();

% compute training performance
results_training = compute_metrics(lx_training_dataset.LxObs,lx_training_dataset.LxNewEquationPred,algorithm_names(4), results_training);
validation_results = struct();
result_trained_model.new_model_equation = new_model_equation;
result_trained_model.new_model_equation.validation_results = validation_results;
result_trained_model.new_model_equation.validation_results.validation_predictions = lx_training_dataset.LxNewEquationPred;
result_trained_model.new_model_equation.validation_results.metrics = results_training("LxNewEquation",:);

% compute test performance
results_test = compute_metrics(lx_test_dataset.LxObs,lx_test_dataset.LxNewEquationPred,algorithm_names(4), results_test);
test_results = struct();
result_trained_model.new_model_equation.test_results = test_results;
result_trained_model.new_model_equation.test_results.test_predictions = lx_test_dataset.LxNewEquationPred;
result_trained_model.new_model_equation.test_results.metrics = results_test("LxNewEquation",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.new_model_equation.test_results.test_predictions,pwbTable,algorithm_names(4),pwbX);


clc;
disp(results_training(:,{'RMSE','R2','Corr Coeff'}));
disp(results_test(:,{'RMSE','R2','Corr Coeff'}));
disp(pwbTable);

writetable(results_training, '1-Trained-Models\Po-all-branches\all-branches-merged\Results-training-model.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models\Po-all-branches\all-branches-merged\Results-test-model.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "1-Trained-Models\Po-all-branches\all-branches-merged\pwbTable.xlsx", "WriteRowNames", true);
save("1-Trained-Models\Po-all-branches\all-branches-merged\Trained_Test_Model.mat","result_trained_model");