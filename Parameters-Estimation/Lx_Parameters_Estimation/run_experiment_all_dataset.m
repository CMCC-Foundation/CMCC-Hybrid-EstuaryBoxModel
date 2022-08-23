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
addpath(genpath('1_Trained-Models'));

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
max_objective_evaluations = 30;

%% Removed useless features
training_dataset = lx_dataset;
training_dataset = removevars(training_dataset, {'DATE','Dataset_Type'});

%% Create table for k-fold cross validation results
algorithm_names = {'regression_trees', 'support_vector_machines', 'gam',...
    'random_forest', 'lsboost', 'neural_network' };

results = table('Size', [6 3], ...
    'VariableTypes', {'double','double','double'}, ...
    'VariableNames', {'k_3_RMSE', 'k_4_RMSE','k_5_RMSE'},...
    'RowNames', algorithm_names);

i=1;
for k = 3:5
    
    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_trees] = regression_tree_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results(1,i) = {results_trees.rmse};

    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_svm] = regression_svm_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results(2,i) = {results_svm.rmse};

    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_gam] = gam_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results(3,i) = {results_gam.rmse};

    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(4), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_rf] = random_forest_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results(4,i) = {results_rf.rmse};

    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(5), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_lsb] = lsboost_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results(5,i) = {results_lsb.rmse};

    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using ", algorithm_names(6), " with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    [results_nn] = neural_network_function(training_dataset,targetFeatureName,1,3,10,50, max_objective_evaluations, k);
    results(6,i) = {results_nn.rmse};


    if k == 3
        compact_struct_trained_model.k_3.regression_trees = results_trees;
        compact_struct_trained_model.k_3.svm = results_svm;
        compact_struct_trained_model.k_3.gam = results_gam;
        compact_struct_trained_model.k_3.random_forest = results_rf;
        compact_struct_trained_model.k_3.lsboost = results_lsb;
        compact_struct_trained_model.k_3.neural_network = results_nn;
    elseif k == 4
        compact_struct_trained_model.k_4.regression_trees = results_trees;
        compact_struct_trained_model.k_4.svm = results_svm;
        compact_struct_trained_model.k_4.gam = results_gam;
        compact_struct_trained_model.k_4.random_forest = results_rf;
        compact_struct_trained_model.k_4.lsboost = results_lsb;
        compact_struct_trained_model.k_4.neural_network = results_nn;
    elseif k == 5
        compact_struct_trained_model.k_5.regression_trees = results_trees;
        compact_struct_trained_model.k_5.svm = results_svm;
        compact_struct_trained_model.k_5.gam = results_gam;
        compact_struct_trained_model.k_5.random_forest = results_rf;
        compact_struct_trained_model.k_5.lsboost = results_lsb;
        compact_struct_trained_model.k_5.neural_network = results_nn;
    end

   
    i = i + 1;

    clc;
    close all;
end

writetable(results, '1-Trained-Models/Results-tuning-with-different-k.xlsx', 'WriteRowNames',true);