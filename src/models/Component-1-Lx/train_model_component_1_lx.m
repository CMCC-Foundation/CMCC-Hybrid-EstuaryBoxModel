% The following script is devoted to the developemnt of 
% Component-1 of Hybrid-EBM. This component is related to the prediction of
% lenght of salinity intrusion (Lx). The aim is to train two machine 
% learning models (Random Forest, LSBoost). 

%% Add path and directory
addpath(genpath("..\..\lib\config\"));
load_path_hyb_ebm_model();

%% Read dataset
lx_dataset = import_dataset("raw\Component-1-Lx\Lx-Dataset.xlsx", 7, "A2:G26", "Lx_obs", ...
    ["Date","Qll", "Qriver", "Sll", "Qtide", "LxObs", "EbmPred"], ...
    ["datetime", "double", "double", "double", "double","double", "double"]);

%% Split dataset in training and test
[lx_training_dataset, lx_test_dataset] = create_training_test_dataset (lx_dataset, 0.2);

%% Save training and test dataset
writetable(lx_training_dataset, '..\..\..\data\processed\Component-1-Lx\Lx-Training-Dataset.xlsx', 'WriteRowNames',true);
writetable(lx_test_dataset, '..\..\..\data\processed\Component-1-Lx\Lx-Test-Dataset.xlsx', 'WriteRowNames',true);

%% Create table for stroring training and test results
algorithm_names = {'EBM','RF','LSBoost'};
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

component_1_trained_models = struct();

%% Compute metrics on EBM (Fully-Phisics model)
EBM = struct();

% compute training performance
results_training = compute_metrics(lx_training_dataset.LxObs,lx_training_dataset.EbmPred, algorithm_names(1), results_training);
validation_results = struct();
component_1_trained_models.EBM = EBM;
component_1_trained_models.EBM.validation_results = validation_results;
component_1_trained_models.EBM.validation_results.validation_predictions = lx_training_dataset.EbmPred;
component_1_trained_models.EBM.validation_results.metrics = results_training("EBM",:);

% compute test performance
results_test = compute_metrics(lx_test_dataset.LxObs,lx_test_dataset.EbmPred,algorithm_names(1), results_test);
test_results = struct();
component_1_trained_models.EBM.test_results = test_results;
component_1_trained_models.EBM.test_results.test_predictions = lx_test_dataset.EbmPred;
component_1_trained_models.EBM.test_results.metrics = results_test("EBM",:);
pwbTable = create_pwb_table(lx_test_dataset.LxObs, ...
    component_1_trained_models.EBM.test_results.test_predictions, ...
    pwbTable, algorithm_names(1), pwbX);

%% Machine learning setting 
% Set target feature for the machine and deep learning model
targetFeatureName = 'LxObs';

% Set maxObjectiveEvaluations as maximum number of objective functions to
% be evaluated in the optimization process
max_objective_evaluations = 60;

% Set k to be used in the k-fold cross validation procedure
k = 5;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
component_1_trained_models.RF = random_forest(...
    lx_training_dataset(:,["Qll","Qriver","Qtide","Sll","LxObs"]), ...
    targetFeatureName, max_objective_evaluations, k);

results_training = compute_metrics(lx_training_dataset.LxObs,...
    component_1_trained_models.RF.validation_results.validation_predictions, ...
    algorithm_names(2), results_training);

component_1_trained_models.RF.validation_results.metrics = results_training("RF",:);
plot_importance(component_1_trained_models.RF.feature_importance, ...
    "Features importance for Lx estimation with Random Forest");
saveas(gcf,"..\..\..\reports\figures\Component-1-Lx\Component-1-Lx-Features-Importance-RF.png");

% save test results
component_1_trained_models.RF.test_results.test_predictions = ...
    component_1_trained_models.RF.model.predictFcn(lx_test_dataset(:,["Qll","Qriver","Qtide","Sll"]));

results_test = compute_metrics( lx_test_dataset.LxObs, ...
    component_1_trained_models.RF.test_results.test_predictions, ...
    algorithm_names(2), results_test);

component_1_trained_models.RF.test_results.metrics = results_test("RF",:);
pwbTable = create_pwb_table(lx_test_dataset.LxObs, ...
    component_1_trained_models.RF.test_results.test_predictions, ...
    pwbTable,algorithm_names(2), pwbX);

%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

component_1_trained_models.LSBoost = lsboost( ...
    lx_training_dataset(:,["Qll","Qriver","Qtide","Sll","LxObs"]), ...
    targetFeatureName, max_objective_evaluations, k);

results_training = compute_metrics(lx_training_dataset.LxObs, ...
    component_1_trained_models.LSBoost.validation_results.validation_predictions, ...
    algorithm_names(3), results_training);

component_1_trained_models.LSBoost.validation_results.metrics = results_training("LSBoost",:);
plot_importance(component_1_trained_models.LSBoost.feature_importance, ...
    "Features importance for Lx estimation with Lsboost");
saveas(gcf,"..\..\..\reports\figures\Component-1-Lx\Component-1-Lx-Features-Importance-LSBoost.png");

% save test results
component_1_trained_models.LSBoost.test_results.test_predictions = ...
    component_1_trained_models.LSBoost.model.predictFcn(lx_test_dataset(:,["Qll","Qriver","Qtide","Sll"]));

results_test = compute_metrics( lx_test_dataset.LxObs, ...
    component_1_trained_models.LSBoost.test_results.test_predictions, ...
    algorithm_names(3), results_test);

component_1_trained_models.LSBoost.test_results.metrics = results_test("LSBoost",:);
pwbTable = create_pwb_table(lx_test_dataset.LxObs, ...
    component_1_trained_models.LSBoost.test_results.test_predictions, ...
    pwbTable, algorithm_names(3), pwbX);

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
    lx_training_dataset.LxObs ...
    lx_training_dataset.EbmPred ...
    component_1_trained_models.RF.validation_results.validation_predictions ...
    component_1_trained_models.LSBoost.validation_results.validation_predictions...
],"VariableNames",{'real_lx','ebm_pred', 'rf_pred', 'lsb_pred'});

create_component_1_4_results_plot(training_table_results,algorithm_names, true, ...
    30, 40, targetFeatureName, "Training",1);

% Test dataset results
test_table_results = array2table([ ...
    lx_test_dataset.LxObs ...
    lx_test_dataset.EbmPred ...
    component_1_trained_models.RF.test_results.test_predictions...
    component_1_trained_models.LSBoost.test_results.test_predictions...
],"VariableNames",{'real_lx', 'ebm_pred', 'rf_pred', 'lsb_pred'});

create_component_1_4_results_plot(test_table_results,algorithm_names, true, ...
    30, 40, targetFeatureName, "Test", 1);

%% Save results and trained models
writetable(results_training, '..\..\..\models\Component-1-Lx\Component-1-Lx-Training-Results.xlsx', 'WriteRowNames',true);
writetable(results_test, '..\..\..\models\Component-1-Lx\Component-1-Lx-Test-Results.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "..\..\..\models\Component-1-Lx\Component-1-Lx-PwbTable.xlsx", "WriteRowNames", true);
save("..\..\..\models\Component-1-Lx\Component-1-Lx-Models.mat","component_1_trained_models");