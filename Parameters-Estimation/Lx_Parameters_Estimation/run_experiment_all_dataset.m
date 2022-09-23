%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with 3 <= k <= 5. 
%  In the end, some figures are plotted and dataset and model are saved. 

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
save('0-Dataset/LX_OBS_WITH_FEATURES.mat', ...
    'lx_dataset');

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Lx_OBS';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 60;

%% Removed useless features
training_dataset = lx_dataset;

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost' };

results = table('Size', [2 3], ...
    'VariableTypes', {'double','double','double'}, ...
    'VariableNames', {'k_3_RMSE', 'k_4_RMSE','k_5_RMSE'},...
    'RowNames', algorithm_names);

i=1;
for k = 3:5
    
    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_rf] = random_forest_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
    results(1,i) = {computeRMSE(training_dataset(:,"Lx_OBS"),results_rf.validation_results.validation_predictions)};

    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_lsb] = lsboost_function(removevars(training_dataset, {'DATE','Dataset_Type'}),targetFeatureName,max_objective_evaluations, k);
    results(2,i) = {computeRMSE(training_dataset(:,"Lx_OBS"),results_lsb.validation_results.validation_predictions)};

    if k == 3
        compact_struct_trained_model.k_3.random_forest = results_rf;
        compact_struct_trained_model.k_3.lsboost = results_lsb;
    elseif k == 4
        compact_struct_trained_model.k_4.random_forest = results_rf;
        compact_struct_trained_model.k_4.lsboost = results_lsb;
    elseif k == 5
        compact_struct_trained_model.k_5.random_forest = results_rf;
        compact_struct_trained_model.k_5.lsboost = results_lsb;
    end
   
    i = i + 1;

    clc;
    close all;
end

writetable(results, '1-Trained-Models/Results-tuning-with-different-k.xlsx', 'WriteRowNames',true);


function [rmse] = computeRMSE(obs, pred)
    if istable(obs)
        obs = table2array(obs);
    end

    if istable(pred)
        pred = table2array(pred);
    end
    rmse = sqrt(sum((obs - pred).^2)/numel(obs));
end