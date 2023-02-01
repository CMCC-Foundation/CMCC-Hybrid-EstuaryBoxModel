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
nVars = 7;
dataRange = "A2:G86";
sheetName = "ALL_BRANCHES";
varNames = ["DateObs","Qocean", "Qriver", "Qtide", "Sll", "LxObs", "Branch"]; 
varTypes = ["datetime", "double", "double", "double", "double", "double", "categorical"];

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
     removevars(lx_training_dataset,{'DateObs', 'Branch'}),...
     removevars(lx_test_dataset,{'DateObs', 'Branch'}));

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

results_training = table('Size', [numel(algorithm_names) 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test = table('Size', [numel(algorithm_names) 8], ...
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
result_trained_model.random_forest = random_forest_function(removevars(lx_training_dataset, {'DateObs', 'Branch'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for Lx estimation with Random Forest");

% save test results
result_trained_model.random_forest.test_results.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(lx_test_dataset, {'DateObs', 'Branch'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.metrics = results_test("random_forest",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.test_predictions,pwbTable,algorithm_names(1),pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

result_trained_model.lsboost = lsboost_function(removevars(lx_training_dataset, {'DateObs','Branch'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for Lx estimation with Lsboost");

% save test results
result_trained_model.lsboost.test_results.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_dataset, {'DateObs', 'Branch'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.metrics = results_test("lsboost",:);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.test_predictions,pwbTable,algorithm_names(2),pwbX);

%% Store predictions in training and test dataset
lx_training_dataset.RandomForest_Prediction = result_trained_model.random_forest.validation_results.validation_predictions;
lx_training_dataset.Lsboost_Prediction = result_trained_model.lsboost.validation_results.validation_predictions;
lx_training_dataset.DatasetType = repmat('Training_Dataset',height(lx_training_dataset),1);

lx_test_dataset.RandomForest_Prediction = result_trained_model.random_forest.test_results.test_predictions;
lx_test_dataset.Lsboost_Prediction = result_trained_model.lsboost.test_results.test_predictions;
lx_test_dataset.DatasetType = repmat('Test_Dataset',height(lx_test_dataset),1);

%% Display performances
clc;
disp(results_training(:,{'RMSE','R2','Corr Coeff'}));
disp(results_test(:,{'RMSE','R2','Corr Coeff'}));
disp(pwbTable);

%% Save results
writetable(lx_training_dataset, '0-Dataset\Po-all-branches\all-branches-merged\lx_training_dataset.xlsx');
writetable(lx_test_dataset, '0-Dataset\Po-all-branches\all-branches-merged\lx_test_dataset.xlsx');
writetable(results_training, '1-Trained-Models\Po-all-branches\all-branches-merged\Results-training-model.xlsx', 'WriteRowNames',true);
writetable(results_test, '1-Trained-Models\Po-all-branches\all-branches-merged\Results-test-model.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "1-Trained-Models\Po-all-branches\all-branches-merged\pwbTable.xlsx", "WriteRowNames", true);
save("1-Trained-Models\Po-all-branches\all-branches-merged\Trained_Test_Model.mat","result_trained_model");
save("0-Dataset\Po-all-branches\all-branches-merged\lx_training_dataset.mat","lx_training_dataset");
save("0-Dataset\Po-all-branches\all-branches-merged\lx_test_dataset.mat","lx_test_dataset");