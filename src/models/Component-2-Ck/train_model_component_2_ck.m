% The following script is devoted to the developemnt of 
% Component-2 of Hybrid-EBM. This component is related to the prediction of
% Ck coefficient. The aim is to train two machine 
% learning models (Random Forest, LSBoost).  

%% Add path and directory
addpath(genpath("..\..\lib\config\"));
load_path_hyb_ebm_model();

%% Read dataset
ck_dataset_rf = import_dataset("\raw\Component-2-Ck\Ck-Obs-RF\Ck-Dataset-RF.xlsx", 6, "A2:F1462", ...
    "Sheet1", ["Year","Doy","Qriver", "Qtide", "Socean", "CkObs"], ...
    ["int16","int16","double", "double","double","double"]);

ck_dataset_lsboost = import_dataset("\raw\Component-2-Ck\Ck-Obs-LSBoost\Ck-Dataset-LSBoost.xlsx", 6, "A2:F1462", ...
    "Sheet1", ["Year","Doy","Qriver", "Qtide", "Socean", "CkObs"], ...
    ["int16","int16","double", "double","double","double"]);

%% Remove missed data
ck_dataset_rf = remove_missing_data_features(ck_dataset_rf);
ck_dataset_lsboost = remove_missing_data_features(ck_dataset_lsboost);

%% Split dataset in training and test
[ck_training_dataset_rf, ck_test_dataset_rf] = create_training_test_dataset (ck_dataset_rf, 0.2);
[ck_training_dataset_lsboost, ck_test_dataset_lsboost] = create_training_test_dataset (ck_dataset_lsboost, 0.2);

%% Save training and test dataset
writetable(ck_training_dataset_rf, '..\..\..\data\processed\Component-2-Ck\Ck-Obs-RF\Ck-Training-Dataset-RF.xlsx', 'WriteRowNames',true);
writetable(ck_test_dataset_rf, '..\..\..\data\processed\Component-2-Ck\Ck-Obs-RF\Ck-Test-Dataset-RF.xlsx', 'WriteRowNames',true);
writetable(ck_training_dataset_lsboost, '..\..\..\data\processed\Component-2-Ck\Ck-Obs-LSBoost\Ck-Training-Dataset-LSBoost.xlsx', 'WriteRowNames',true);
writetable(ck_test_dataset_lsboost, '..\..\..\data\processed\Component-2-Ck\Ck-Obs-LSBoost\Ck-Test-Dataset-LSBoost.xlsx', 'WriteRowNames',true);

%% Create table for stroring training and test results
algorithm_names = {'RF', 'LSBoost'};
metrics_names = {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'};
pwbX = [1 5 10 20 30];

pwbTable = table('Size',[numel(pwbX) numel(algorithm_names)],...
    'VariableTypes', repmat({'double'}, 1, numel(algorithm_names)), ...
    'VariableNames', algorithm_names,...
    'RowNames', strcat('PWB',string(pwbX)));

results_training = table('Size', [numel(algorithm_names) numel(metrics_names)], ...
    'VariableTypes', repmat({'double'}, 1, numel(metrics_names)), ...
    'VariableNames', metrics_names,...
    'RowNames', algorithm_names);

results_test = table('Size', [numel(algorithm_names) numel(metrics_names)], ...
    'VariableTypes', repmat({'double'}, 1, numel(metrics_names)), ...
    'VariableNames', metrics_names,...
    'RowNames', algorithm_names);

component_2_trained_models = struct();

%% Machine learning setting 
% Set target feature for the machine and deep learning model
targetFeatureName = 'CkObs';

% Set maxObjectiveEvaluations as maximum number of objective functions to
% be evaluated in the optimization process
max_objective_evaluations = 60;

% Set k to be used in the k-fold cross validation procedure
k = 5;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
component_2_trained_models.random_forest = random_forest( ...
    ck_training_dataset_rf(:,["Qriver","Qtide","Socean","CkObs"]), ...
    targetFeatureName, max_objective_evaluations, k);

component_2_trained_models.random_forest.validation_results.validation_predictions...
    (component_2_trained_models.random_forest.validation_results.validation_predictions < 0) = 0;

results_training = compute_metrics(ck_training_dataset_rf.CkObs,...
    component_2_trained_models.random_forest.validation_results.validation_predictions, ...
    algorithm_names(1), results_training);

component_2_trained_models.random_forest.validation_results.metrics = results_training("RF",:);
plot_importance(component_2_trained_models.random_forest.feature_importance, ...
    "Features importance for ck estimation with Random Forest");
saveas(gcf,"..\..\..\reports\figures\Component-2-Ck\Component-2-Ck-Features-Importance-RF.png");

% save test results
test_results = struct();
component_2_trained_models.random_forest.test_results = test_results;
component_2_trained_models.random_forest.test_results.test_predictions = ...
    component_2_trained_models.random_forest.model.predictFcn(ck_test_dataset_rf(:,["Qriver","Qtide","Socean"]));

component_2_trained_models.random_forest.test_results.test_predictions...
    (component_2_trained_models.random_forest.test_results.test_predictions < 0) = 0;

results_test= compute_metrics( ck_test_dataset_rf.CkObs, ...
    component_2_trained_models.random_forest.test_results.test_predictions, ...
    algorithm_names(1), results_test);

component_2_trained_models.random_forest.test_results.metrics = results_test("RF",:);
pwbTable = create_pwb_table( ck_test_dataset_rf.CkObs, ...
    component_2_trained_models.random_forest.test_results.test_predictions, ...
    pwbTable, algorithm_names(1), pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
component_2_trained_models.lsboost = lsboost( ...
    ck_training_dataset_lsboost(:,["Qriver","Qtide","Socean","CkObs"]), ...
    targetFeatureName, max_objective_evaluations, k);

component_2_trained_models.lsboost.validation_results.validation_predictions...
    (component_2_trained_models.lsboost.validation_results.validation_predictions < 0) = 0;

results_training = compute_metrics( ck_training_dataset_lsboost.CkObs, ...
    component_2_trained_models.lsboost.validation_results.validation_predictions, ...
    algorithm_names(2), results_training);

component_2_trained_models.lsboost.validation_results.metrics = results_training("LSBoost",:);
plot_importance(component_2_trained_models.lsboost.feature_importance, ...
    "Features importance for ck estimation with Lsboost");
saveas(gcf,"..\..\..\reports\figures\Component-2-Ck\Component-2-Ck-Features-Importance-LSBoost.png");

% save test results
test_results = struct();
component_2_trained_models.lsboost.test_results = test_results;
component_2_trained_models.lsboost.test_results.test_predictions = ...
    component_2_trained_models.lsboost.model.predictFcn(ck_test_dataset_lsboost(:,["Qriver","Qtide","Socean"]));

component_2_trained_models.lsboost.test_results.test_predictions...
    (component_2_trained_models.lsboost.test_results.test_predictions < 0) = 0;

results_test= compute_metrics( ck_test_dataset_lsboost.CkObs, ...
    component_2_trained_models.lsboost.test_results.test_predictions, ...
    algorithm_names(2), results_test);

component_2_trained_models.lsboost.test_results.metrics = results_test("LSBoost",:);
pwbTable = create_pwb_table( ck_test_dataset_lsboost.CkObs, ...
    component_2_trained_models.lsboost.test_results.test_predictions, ...
    pwbTable, algorithm_names(2), pwbX);

%% display results: metrics and pwb-table
clc;
disp("Training results")
disp("--------------------------------------------------------")
disp(results_training(:,["RMSE","MAE","Corr Coeff"]));
disp("Test results")
disp("--------------------------------------------------------")
disp(results_test(:,["RMSE","MAE","Corr Coeff"]));
disp(pwbTable);

%% display results: perfect-fit-plot and response-plot
% Training dataset results
training_table_results = array2table([ ...
    ck_training_dataset_rf.CkObs ...
    component_2_trained_models.random_forest.validation_results.validation_predictions...
    ck_training_dataset_lsboost.CkObs ...
    component_2_trained_models.lsboost.validation_results.validation_predictions...
],"VariableNames",{'real_ck_rf' ,'rf_pred', 'real_ck_lsb','lsb_pred'});

create_component_2_results_plot(training_table_results,algorithm_names, true, ...
    30, 1100, targetFeatureName, "Training");

% Test dataset result
test_table_results = array2table([ ...
    ck_test_dataset_rf.CkObs ...
    component_2_trained_models.random_forest.test_results.test_predictions...
    ck_test_dataset_lsboost.CkObs ...
    component_2_trained_models.lsboost.test_results.test_predictions...
],"VariableNames",{'real_ck_rf' ,'rf_pred', 'real_ck_lsb','lsb_pred'});

create_component_2_results_plot(test_table_results,algorithm_names, true, ...
    30, 1100, targetFeatureName, "Test");

%% Save results and trained models
writetable(results_training, '..\..\..\models\Component-2-Ck\Component-2-Ck-Training-Results.xlsx', 'WriteRowNames',true);
writetable(results_test, '..\..\..\models\Component-2-Ck\Component-2-Ck-Test-Results.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "..\..\..\models\Component-2-Ck\Component-2-Ck-PwbTable.xlsx", "WriteRowNames", true);
save("..\..\..\models\Component-2-Ck\Component-2-Ck-Models.mat","component_2_trained_models");