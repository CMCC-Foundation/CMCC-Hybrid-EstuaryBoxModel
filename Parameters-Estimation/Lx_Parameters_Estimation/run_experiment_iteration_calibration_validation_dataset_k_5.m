%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('1-Trained-Models'));

%% Set import dataset settings
filepath = "0-Dataset\LX_OBS_WITH_FEATURES.xlsx";
nVars = 7;
dataRange = "A2:G26";
sheetName = "Lx_obs";
varNames = ["DATE","Q_l", "Q_r", "S_l", "Q_tide", "Lx_OBS", "Dataset_Type"]; 
varTypes = ["datetime", "double", "double", "double", "double","double","categorical"];

[lx_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Define number of iteration
niteration = 5;
iterationLabel = {'Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5'};

%% Create tables to store results
% random forest training
results_training_rf = table('Size', [niteration 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

% random forest test
results_test_rf = table('Size', [niteration 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

% lsboost training
results_training_lsb = table('Size', [niteration 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

% lsboost test
results_test_lsb = table('Size', [niteration 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Lx_OBS';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 60;

percetageSplitData = 0.20 ;
k = 4;

%% Run experiment
for i = 1: niteration
    
    [training_dataset, testing_dataset] = create_training_test_dataset(lx_dataset, percetageSplitData);

    %% Training random forest model
    fprintf("\n===================================================================\n");
    fprintf(strcat("Iteration n. ",string(i)));
    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using random forest with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    
    model_rf = random_forest_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
    results_training_rf = compute_metrics(training_dataset(:,targetFeatureName),model_rf.validation_results.validation_predictions, iterationLabel(i), results_training_rf);

    %% Testing random forest
    predictions_rf = model_rf.model.predictFcn(removevars(testing_dataset, {'DATE','Dataset_Type'}));
    results_test_rf = compute_metrics(testing_dataset(:,targetFeatureName), predictions_rf,iterationLabel(i),results_test_rf);

    %% Training lsboost model
    fprintf("\n===================================================================\n");
    fprintf(strcat("Iteration n. ",string(i)));
    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using lsboost with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    
    model_lsb = lsboost_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
    results_training_lsb = compute_metrics(training_dataset(:,targetFeatureName),model_lsb.validation_results.validation_predictions, iterationLabel(i), results_training_lsb);
    
    %% Testing lsboost
    predictions_lsb = model_lsb.model.predictFcn(removevars(testing_dataset, {'DATE','Dataset_Type'}));
    results_test_lsb = compute_metrics(testing_dataset(:,targetFeatureName), predictions_lsb,iterationLabel(i),results_test_lsb);

    clc;
    close all;
end

writetable(results_training_rf,'1-Trained-Models/Iteration-Results/training_rf.xlsx','WriteRowNames',true);
writetable(results_training_lsb,'1-Trained-Models/Iteration-Results/training_lsb.xlsx','WriteRowNames',true);
writetable(results_test_rf,'1-Trained-Models/Iteration-Results/test_rf.xlsx','WriteRowNames',true);
writetable(results_test_lsb,'1-Trained-Models/Iteration-Results/test_lsb.xlsx','WriteRowNames',true);
results_iteration = struct("results_training_lsb",results_training_lsb,"results_training_rf",results_training_rf,...
    "results_test_lsb",results_test_lsb,"results_test_rf",results_test_rf);
save("1-Trained-Models\Iteration-Results\results-iteration.mat", "results_iteration");


figure;
subplot(2,2,1);
plotPerformance(table2array(results_training_rf), 'Training: Random forest');

subplot(2,2,2);
plotPerformance(table2array(results_training_lsb), 'Training: Lsboost');

subplot(2,2,3);
plotPerformance(table2array(results_test_rf), 'Test: Random forest');

subplot(2,2,4);
plotPerformance(table2array(results_test_lsb), 'Test: Lsboost');

function [] = plotPerformance(dataset, titlePlot)
    plot(dataset, '--.', 'MarkerSize',18,'MarkerEdgeColor','auto', 'LineWidth',1);
    xticks([1 2 3 4 5]);
    xticklabels({'1','2','3','4','5'});

    xlim([0.9 5.1]);

    xlabel('Iterarion');
    ylabel('Performance');
    legend({'RMSE','NRMSE','MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'});
    title(titlePlot);
end