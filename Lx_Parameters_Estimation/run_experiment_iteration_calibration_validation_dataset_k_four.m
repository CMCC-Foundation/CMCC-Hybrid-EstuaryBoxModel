%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('1-Pre-Processing'));
addpath(genpath('2-Machine-Learning-Function'));
addpath(genpath('3_Trained-Models'));
addpath(genpath('3_Trained-Models/Iteration-Results'));

filepath = "0-Dataset\LX_OBS_WITH_FEATURES.xlsx";
[lx_dataset] = import_dataset(filepath);

%% Removing useless features 
lx_dataset = removevars(lx_dataset, {'DATE','Dataset_Type'});

%% Define number of iteration
niteration = 5;
iterationLabel = {'Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5'};

%% Create tables to store results
% random forest training
results_training_rf = table('Size', [niteration 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

% random forest test
results_test_rf = table('Size', [niteration 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

% lsboost training
results_training_lsb = table('Size', [niteration 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

% lsboost test
results_test_lsb = table('Size', [niteration 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', iterationLabel);

%% Set target feature for the machine and deep learning model
targetFeatureName = 'Lx_OBS';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 30;

percetageSplitData = 0.80 ;
k = 4;

%% Run experiment
for i = 1: niteration
    
    [training_dataset, testing_dataset] = generateTrainingTest(lx_dataset, percetageSplitData);

    %% Training random forest model
    fprintf("\n===================================================================\n");
    fprintf(strcat("Iteration n. ",string(i)));
    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using random forest with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    
    model_rf = random_forest_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results_training_rf(i,"RMSE") = {model_rf.metrics.rmse};
    results_training_rf(i,"MAE") = {computeMAE(training_dataset.Lx_OBS, model_rf.predictions)};
    results_training_rf(i,"RSE") = {computeRSE(training_dataset.Lx_OBS, model_rf.predictions)};
    results_training_rf(i,"RRSE") = {sqrt(table2array(results_training_rf(i,"RSE")))};
    results_training_rf(i,"RAE") = {computeRAE(training_dataset.Lx_OBS, model_rf.predictions)};
    results_training_rf(i,"R2") = {computeR2(training_dataset.Lx_OBS, model_rf.predictions)};
    results_training_rf(i,"Corr Coeff") = {computeCorrCoef( training_dataset.Lx_OBS, model_rf.predictions)};
    
    %% Testing random forest
    predictions_rf = model_rf.model.predictFcn(testing_dataset);
    results_test_rf(i,"RMSE") = {computeRMSE(testing_dataset.Lx_OBS, predictions_rf)};
    results_test_rf(i,"MAE") = {computeMAE(testing_dataset.Lx_OBS, predictions_rf)};
    results_test_rf(i,"RSE") = {computeRSE(testing_dataset.Lx_OBS, predictions_rf)};
    results_test_rf(i,"RRSE") = {sqrt(table2array(results_test_rf(i,"RSE")))};
    results_test_rf(i,"RAE") = {computeRAE(testing_dataset.Lx_OBS, predictions_rf)};
    results_test_rf(i,"R2") = {computeR2(testing_dataset.Lx_OBS, predictions_rf)};
    results_test_rf(i,"Corr Coeff") = {computeCorrCoef(testing_dataset.Lx_OBS, predictions_rf)};

    %% Training lsboost model
    fprintf("\n===================================================================\n");
    fprintf(strcat("Iteration n. ",string(i)));
    fprintf("\n===================================================================\n");
    fprintf(strcat("Training model using lsboost with k=", string(k), "\n"));
    fprintf("===================================================================\n");
    
    model_lsb = lsboost_function(training_dataset,targetFeatureName,max_objective_evaluations, k);
    results_training_lsb(i,"RMSE") = {model_lsb.metrics.rmse};
    results_training_lsb(i,"MAE") = {computeMAE(training_dataset.Lx_OBS, model_lsb.predictions)};
    results_training_lsb(i,"RSE") = {computeRSE(training_dataset.Lx_OBS, model_lsb.predictions)};
    results_training_lsb(i,"RRSE") = {sqrt(table2array(results_training_lsb(i,"RSE")))};
    results_training_lsb(i,"RAE") = {computeRAE(training_dataset.Lx_OBS, model_lsb.predictions)};
    results_training_lsb(i,"R2") = {computeR2(training_dataset.Lx_OBS, model_lsb.predictions)};
    results_training_lsb(i,"Corr Coeff") = {computeCorrCoef( training_dataset.Lx_OBS, model_lsb.predictions)};
    
    %% Testing lsboost
    predictions_lsb = model_lsb.model.predictFcn(testing_dataset);
    results_test_lsb(i,"RMSE") = {computeRMSE(testing_dataset.Lx_OBS, predictions_lsb)};
    results_test_lsb(i,"MAE") = {computeMAE(testing_dataset.Lx_OBS, predictions_lsb)};
    results_test_lsb(i,"RSE") = {computeRSE(testing_dataset.Lx_OBS, predictions_lsb)};
    results_test_lsb(i,"RRSE") = {sqrt(table2array(results_test_lsb(i,"RSE")))};
    results_test_lsb(i,"RAE") = {computeRAE(testing_dataset.Lx_OBS, predictions_lsb)};
    results_test_lsb(i,"R2") = {computeR2(testing_dataset.Lx_OBS, predictions_lsb)};
    results_test_lsb(i,"Corr Coeff") = {computeCorrCoef(testing_dataset.Lx_OBS, predictions_rf)};
    
    disp(string(testing_dataset.Lx_OBS));

    clc;
    close all;
end

writetable(results_training_rf,'3-Trained-Models/Iteration-Results/training_rf.xlsx');
writetable(results_training_lsb,'3-Trained-Models/Iteration-Results/training_lsb.xlsx');
writetable(results_test_rf,'3-Trained-Models/Iteration-Results/test_rf.xlsx');
writetable(results_test_lsb,'3-Trained-Models/Iteration-Results/test_lsb.xlsx');

figure;
subplot(2,2,1);
plotPerformance(table2array(results_training_rf), 'Training: Random forest');

subplot(2,2,2);
plotPerformance(table2array(results_training_lsb), 'Training: Lsboost');

subplot(2,2,3);
plotPerformance(table2array(results_test_rf), 'Test: Random forest');

subplot(2,2,4);
plotPerformance(table2array(results_test_lsb), 'Test: Lsboost');


function [training, test] = generateTrainingTest(dataset, percetageSplitData)
    % Split randomly in training and test
    nSample = height(dataset);
    rng('shuffle');
    idx = randperm(nSample)  ;
    training = dataset(idx(1:round(percetageSplitData*nSample)),:) ; 
    test = dataset(idx(round(percetageSplitData*nSample)+1:end),:) ;
end

function [rmse] = computeRMSE(obs, pred)
    rmse = sqrt(sum((obs - pred).^2)/height(obs));
end

function [mae] = computeMAE(obs, pred)
    mae = (sum(abs(pred-obs)))/height(obs);
end

function [rse] = computeRSE (obs, pred)
    num = sum((pred-obs).^2);
    den = sum((obs-mean(obs)).^2);
    rse = num/den;
end

function [rae] = computeRAE (obs, pred)
    num = sum(abs(pred-obs));
    den = sum(abs(mean(obs) - obs));
    rae = num / den;
end

function [r2] = computeR2 (obs, pred)
    sse = sum((obs-pred).^2);
    sst = sum((obs - mean(obs)).^2);
    r2 = 1 - (sse/sst);
end

function [r] = computeCorrCoef(obs, pred)
    corr_coeff_matrix = corrcoef(obs, pred);
    r = corr_coeff_matrix(1,2);
end


function [] = plotPerformance(dataset, titlePlot)
    plot(dataset, '--.', 'MarkerSize',18,'MarkerEdgeColor','auto');
    xticks([1 2 3 4 5]);
    xticklabels({'1','2','3','4','5'});

    xlim([0.9 5.1]);

    xlabel('Iterarion');
    ylabel('Performance');
    legend({'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'});
    title(titlePlot);
end