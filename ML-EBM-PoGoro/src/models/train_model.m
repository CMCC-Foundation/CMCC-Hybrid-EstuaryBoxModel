clc
clear
close all

addpath(genpath("..\lib\"));
addpath(genpath("..\visualization\"));
addpath(genpath("..\..\data\"));

%% Read dataset
sul_dataset = import_dataset("processed\input-features-sul.xlsx", 11, "A2:K1462", "Sheet1", ...
    ["ID","Date","Qriver", "Qll", "Qtidef", "Sll", "Socean", "Sul", "Sul_EBM", "Dataset", "Season"], ...
    ["categorical","datetime", "double", "double", "double", "double","double", "double", "double", "categorical", "categorical"]);

%% Split in training and test
sul_training = sul_dataset(sul_dataset.Dataset == "Training",:);
sul_test = sul_dataset(sul_dataset.Dataset == "Test",:);

%% Save training and test dataset
%writetable(sul_training, "..\..\data\processed\training-dataset-sul.xlsx");
%writetable(sul_test, "..\..\data\processed\test-dataset-sul.xlsx");

%% Create table for stroring training and test results
algorithm_names = {'EBM','RF','LSBoost','NN'};
metrics_names = {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'};
pwbX = [1 5 10 20 30];

pwbTable = table('Size', [numel(pwbX) numel(algorithm_names)],...
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

% Struct to store the trained models
trained_models = struct();

% Table to store training predictions
models_training_predictions = table();
models_training_predictions.ID = sul_training.ID;
models_training_predictions.Date = sul_training.Date;
models_training_predictions.Sul = sul_training.Sul;

% Table to store test predictions
models_test_predictions = table();
models_test_predictions.ID = sul_test.ID;
models_test_predictions.Date = sul_test.Date;
models_test_predictions.Sul = sul_test.Sul;

%% Compute metrics on EBM (Fully-Phisics model)
% compute training performance
results_training = compute_metrics(sul_training.Sul, sul_training.Sul_EBM, algorithm_names(1), results_training);
models_training_predictions.Sul_EBM = sul_training.Sul_EBM;

% compute test performance
results_test = compute_metrics(sul_test.Sul,sul_test.Sul_EBM,algorithm_names(1), results_test);
models_test_predictions.Sul_EBM = sul_test.Sul_EBM;
pwbTable = create_pwb_table(sul_test.Sul, sul_test.Sul_EBM, ...
    pwbTable, algorithm_names(1), pwbX);

%% Experimental setting definition
% Retrive predictors and response name for the machine learning model
predictors = ["Qriver", "Qtidef", "Socean", "Sll", "Qll", "Season"];
response = "Sul";

% Set maxObjectiveEvaluations as maximum number of objective functions to
% be evaluated in the optimization process
max_objective_evaluations = 150;

% Set k to be used in the k-fold cross validation procedure
k = 5; 

%% Random forest algorithm
fprintf("\n===================================================================\n");
fprintf(strcat("Training model ", algorithm_names(2),"\n"));
fprintf("===================================================================\n");

% Define the research space for hyperparameters
params_settings = define_optimizable_variable_ensemble_method( ...
    sul_training(:,predictors), sul_training(:,response), ...
    [10, 1000], 0, [1, 1000], [1, 2500], [1 5]);

% Train the random forest model
[trained_model, training_predicrions] = ensemble_method(sul_training, ...
    predictors, response, max_objective_evaluations, k, 'Bag', params_settings);

% Save the trained model and training predictions
trained_models.RF = trained_model;
models_training_predictions.Sul_RF = training_predicrions;
models_training_predictions.Sul_RF(models_training_predictions.Sul_RF < 0) = 0;

% Evaluate the training performance
results_training = compute_metrics(sul_training.Sul, models_training_predictions.Sul_RF, ...
    algorithm_names(2), results_training);

% Predict on test set
models_test_predictions.Sul_RF = trained_models.RF.model.predictFcn(sul_test(:, predictors));
models_test_predictions.Sul_RF(models_test_predictions.Sul_RF < 0) = 0;

% Evaluate the test performance
results_test = compute_metrics(sul_test.Sul, models_test_predictions.Sul_RF, ...
    algorithm_names(2), results_test);
pwbTable = create_pwb_table(sul_test.Sul, models_test_predictions.Sul_RF, ...
    pwbTable, algorithm_names(2), pwbX);

% Plot features importance
f = plot_importance(trained_models.RF.feature_importance, ...
    "Features importance for Sul estimation with RF");
saveas(f, "..\..\reports\figure\Season-Feature\Features-Importance-RF.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Features-Importance-RF.jpg","BackgroundColor","white", "Resolution", 600);

%% LSBoost algorithm
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3),"\n"));
fprintf("===================================================================\n");

% Define the research space for hyperparameters
params_settings = define_optimizable_variable_ensemble_method( ...
    sul_training(:,predictors), sul_training(:,response), ...
    [10, 1000], [1e-04, 1], [1, 1000], [1, 2500], [1 5]);

% Train the lsboost model
[trained_model, training_predicrions] = ensemble_method(sul_training, ...
    predictors, response, max_objective_evaluations, k, 'LSBoost', params_settings);

% Save the trained model and training predictions
trained_models.LSBoost = trained_model;
models_training_predictions.Sul_LSBoost = training_predicrions;
models_training_predictions.Sul_LSBoost(models_training_predictions.Sul_LSBoost < 0) = 0;

% Evaluate the training performance
results_training = compute_metrics(sul_training.Sul, models_training_predictions.Sul_LSBoost, ...
    algorithm_names(3), results_training);

% Predict on test set
models_test_predictions.Sul_LSBoost = trained_models.LSBoost.model.predictFcn(sul_test(:, predictors));
models_test_predictions.Sul_LSBoost(models_test_predictions.Sul_LSBoost < 0) = 0;

% Evaluate the test performance
results_test = compute_metrics(sul_test.Sul, models_test_predictions.Sul_LSBoost, ...
    algorithm_names(3), results_test);
pwbTable = create_pwb_table(sul_test.Sul, models_test_predictions.Sul_LSBoost, ...
    pwbTable, algorithm_names(3), pwbX);

% Plot features importance
f = plot_importance(trained_models.LSBoost.feature_importance, ...
    "Features importance for Sul estimation with LSBoost");
saveas(f, "..\..\reports\figure\Season-Feature\Features-Importance-LSBoost.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Features-Importance-LSBoost.jpg","BackgroundColor","white", "Resolution", 600);

%% Neural Network algorithm
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(4),"\n"));
fprintf("===================================================================\n");

% Define the research space for hyperparameters
params_settings = define_optimizable_variable_nn( ...
    sul_training(:,predictors), sul_training(:,response), ...
    [1, 5], [10, 500], true, true, [1.038e-08,1.0384e+02], true, true);

% Train the neural network model
[trained_model, training_predicrions] = neural_network(sul_training, ...
    predictors, response, max_objective_evaluations, k, params_settings);

% Save the trained model and training predictions
trained_models.NN = trained_model;
models_training_predictions.Sul_NN = training_predicrions;
models_training_predictions.Sul_NN(models_training_predictions.Sul_NN < 0) = 0;

% Evaluate the training performance
results_training = compute_metrics(sul_training.Sul, models_training_predictions.Sul_NN, ...
    algorithm_names(4), results_training);

% Predict on test set
models_test_predictions.Sul_NN = trained_models.NN.model.predictFcn(sul_test(:, predictors));
models_test_predictions.Sul_NN(models_test_predictions.Sul_NN < 0) = 0;

% Evaluate the test performance
results_test = compute_metrics(sul_test.Sul, models_test_predictions.Sul_NN, ...
    algorithm_names(4), results_test);
pwbTable = create_pwb_table(sul_test.Sul, models_test_predictions.Sul_NN, ...
    pwbTable, algorithm_names(4), pwbX);

%% Save results and trained models
writetable(results_training, '..\..\models\Season-Feature\results-summary.xlsx', 'WriteRowNames',true, 'Sheet', 'result-training');
writetable(results_test, '..\..\models\Season-Feature\results-summary.xlsx', 'WriteRowNames',true, 'Sheet', 'result-test');
writetable(pwbTable, '..\..\models\Season-Feature\results-summary.xlsx', "WriteRowNames", true, 'Sheet', "pwb-test");
writetable(models_training_predictions, '..\..\models\Season-Feature\models-predictions.xlsx', 'WriteRowNames',true, 'Sheet', 'training-predictions');
writetable(models_test_predictions, '..\..\models\Season-Feature\models-predictions.xlsx', 'WriteRowNames',true, 'Sheet', 'test-predictions');
save('..\..\models\Season-Feature\trained-models.mat','trained_models');