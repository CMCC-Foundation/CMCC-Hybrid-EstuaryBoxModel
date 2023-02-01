%% The following script group all the experiment carried out in this paper:
%  Given the dataset, some useless features are removed. 
%  After that,  we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 5. 
%  We use 20 examples to train and validate our model, and 5 examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset\Po-all-branches\train-goro-test-gnocca-tolle-dritta'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1-Trained-Models\Po-all-branches\train-goro-test-gnocca-tolle-dritta'));

%% Set import dataset settings
filepath = "0-Dataset\Po-all-branches\train-goro-test-gnocca-tolle-dritta\LX_OBS_ALL_BRANCHES_MERGED.xlsx";
nVars = 10;
dataRange = "A2:J86";
sheetName = "ALL_BRANCHES";
varNames = ["DateObs","Qocean", "Qriver", "Qtide", "Sll", "LxObs", "LxOldEquationPred", "LxNewEquationPred","Branch","DatasetType"]; 
varTypes = ["datetime", "double", "double", "double", "double","double", "double", "double", "categorical", "categorical"];

[lx_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Split original dataset in training and test set
lx_training_dataset = lx_dataset(lx_dataset.DatasetType == "Training",:);
lx_test_gnocca = lx_dataset(lx_dataset.DatasetType == "Test_Gnocca",:);
lx_test_tolle = lx_dataset(lx_dataset.DatasetType == "Test_Tolle",:);
lx_test_dritta = lx_dataset(lx_dataset.DatasetType == "Test_Dritta",:);
lx_test_dataset = [lx_test_gnocca; lx_test_tolle;lx_test_dritta];

%% Set target feature for the machine and deep learning model
targetFeatureName = 'LxObs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 5;

%% Plot boxplot for training and test dataset
plot_boxplot("Boxplot of features for Lx estimation",...
     removevars(lx_training_dataset,{'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}),...
     removevars(lx_test_gnocca,{'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch','DatasetType' }), ...
     removevars(lx_test_tolle,{'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch','DatasetType' }), ...
     removevars(lx_test_dritta,{'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch','DatasetType' }));

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

results_test_gnocca = table('Size', [4 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_tolle = table('Size', [4 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_dritta = table('Size', [4 8], ...
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

%% Random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(lx_training_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",["RMSE","R2","Corr Coeff"]);
plot_importance(result_trained_model.random_forest.feature_importance, "Features importance for Lx estimation with Random Forest");

% save test results on Po Gnocca
Gnocca = struct();
result_trained_model.random_forest.test_results.Gnocca = Gnocca;
result_trained_model.random_forest.test_results.Gnocca.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(lx_test_gnocca, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test_gnocca = compute_metrics(lx_test_gnocca(:,targetFeatureName), result_trained_model.random_forest.test_results.Gnocca.test_predictions, algorithm_names(1), results_test_gnocca);
result_trained_model.random_forest.test_results.Gnocca.metrics = results_test_gnocca("random_forest",["RMSE","R2","Corr Coeff"]);

% save test results on Po Tolle
Tolle = struct();
result_trained_model.random_forest.test_results.Tolle = Tolle;
result_trained_model.random_forest.test_results.Tolle.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(lx_test_tolle, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test_tolle = compute_metrics(lx_test_tolle(:,targetFeatureName), result_trained_model.random_forest.test_results.Tolle.test_predictions, algorithm_names(1), results_test_tolle);
result_trained_model.random_forest.test_results.Tolle.metrics = results_test_tolle("random_forest",["RMSE","R2","Corr Coeff"]);

% save test results on Po Dritta
Dritta = struct();
result_trained_model.random_forest.test_results.Dritta = Dritta;
result_trained_model.random_forest.test_results.Dritta.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(lx_test_dritta, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test_dritta = compute_metrics(lx_test_dritta(:,targetFeatureName), result_trained_model.random_forest.test_results.Dritta.test_predictions, algorithm_names(1), results_test_dritta);
result_trained_model.random_forest.test_results.Dritta.metrics = results_test_dritta("random_forest",["RMSE","R2","Corr Coeff"]);

%% Lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.lsboost = lsboost_function(removevars(lx_training_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch','DatasetType'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(lx_training_dataset(:,targetFeatureName),result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",["RMSE","R2","Corr Coeff"]);
plot_importance(result_trained_model.lsboost.feature_importance, "Features importance for Lx estimation with Lsboost");

% save test results on Po Gnocca
Gnocca = struct();
result_trained_model.lsboost.test_results.Gnocca = Gnocca;
result_trained_model.lsboost.test_results.Gnocca.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_gnocca, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test_gnocca = compute_metrics(lx_test_gnocca(:,targetFeatureName), result_trained_model.lsboost.test_results.Gnocca.test_predictions, algorithm_names(2), results_test_gnocca);
result_trained_model.lsboost.test_results.Gnocca.metrics = results_test_gnocca("lsboost",["RMSE","R2","Corr Coeff"]);

% save test results on Po Tolle
Tolle = struct();
result_trained_model.lsboost.test_results.Tolle = Tolle;
result_trained_model.lsboost.test_results.Tolle.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_tolle, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test_tolle = compute_metrics(lx_test_tolle(:,targetFeatureName), result_trained_model.lsboost.test_results.Tolle.test_predictions, algorithm_names(2), results_test_tolle);
result_trained_model.lsboost.test_results.Tolle.metrics = results_test_tolle("lsboost",["RMSE","R2","Corr Coeff"]);

% save test results on Po Dritta
Dritta = struct();
result_trained_model.lsboost.test_results.Dritta = Dritta;
result_trained_model.lsboost.test_results.Dritta.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_dritta, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test_dritta = compute_metrics(lx_test_dritta(:,targetFeatureName), result_trained_model.lsboost.test_results.Dritta.test_predictions, algorithm_names(2), results_test_dritta);
result_trained_model.lsboost.test_results.Dritta.metrics = results_test_dritta("lsboost",["RMSE","R2","Corr Coeff"]);

%% Compute metrics LxOldEquation
old_model_equation = struct();

% compute training performance 
results_training = compute_metrics(lx_training_dataset.LxObs,lx_training_dataset.LxOldEquationPred,algorithm_names(3), results_training);
validation_results = struct();
result_trained_model.old_model_equation = old_model_equation;
result_trained_model.old_model_equation.validation_results = validation_results;
result_trained_model.old_model_equation.validation_results.validation_predictions = lx_training_dataset.LxOldEquationPred;
result_trained_model.old_model_equation.validation_results.metrics = results_training("LxOldEquation",["RMSE","R2","Corr Coeff"]);

test_results = struct();
result_trained_model.old_model_equation.test_results = test_results;

% compute test performance on Po Gnocca
Gnocca = struct();
results_test_gnocca = compute_metrics(lx_test_gnocca.LxObs,lx_test_gnocca.LxOldEquationPred,algorithm_names(3), results_test_gnocca);
result_trained_model.old_model_equation.test_results.Gnocca = Gnocca;
result_trained_model.old_model_equation.test_results.Gnocca.test_predictions = lx_test_gnocca.LxOldEquationPred;
result_trained_model.old_model_equation.test_results.Gnocca.metrics = results_test_gnocca("LxOldEquation",["RMSE","R2","Corr Coeff"]);

% compute test performance on Po Tolle
Tolle = struct();
results_test_tolle = compute_metrics(lx_test_tolle.LxObs,lx_test_tolle.LxOldEquationPred,algorithm_names(3), results_test_tolle);
result_trained_model.old_model_equation.test_results.Tolle = Tolle;
result_trained_model.old_model_equation.test_results.Tolle.test_predictions = lx_test_tolle.LxOldEquationPred;
result_trained_model.old_model_equation.test_results.Tolle.metrics = results_test_tolle("LxOldEquation",["RMSE","R2","Corr Coeff"]);

% compute test performance on Po Dritta
Dritta = struct();
results_test_dritta = compute_metrics(lx_test_dritta.LxObs,lx_test_dritta.LxOldEquationPred,algorithm_names(3), results_test_dritta);
result_trained_model.old_model_equation.test_results.Dritta = Dritta;
result_trained_model.old_model_equation.test_results.Dritta.test_predictions = lx_test_dritta.LxOldEquationPred;
result_trained_model.old_model_equation.test_results.Dritta.metrics = results_test_dritta("LxOldEquation",["RMSE","R2","Corr Coeff"]);

%% Compute metrics LxNewEquation
new_model_equation = struct();

% compute training performance 
results_training = compute_metrics(lx_training_dataset.LxObs,lx_training_dataset.LxNewEquationPred,algorithm_names(4), results_training);
validation_results = struct();
result_trained_model.new_model_equation = new_model_equation;
result_trained_model.new_model_equation.validation_results = validation_results;
result_trained_model.new_model_equation.validation_results.validation_predictions = lx_training_dataset.LxNewEquationPred;
result_trained_model.new_model_equation.validation_results.metrics = results_training("LxNewEquation",["RMSE","R2","Corr Coeff"]);

test_results = struct();
result_trained_model.new_model_equation.test_results = test_results;

% compute test performance on Po Gnocca
Gnocca = struct();
results_test_gnocca = compute_metrics(lx_test_gnocca.LxObs,lx_test_gnocca.LxNewEquationPred,algorithm_names(4), results_test_gnocca);
result_trained_model.new_model_equation.test_results.Gnocca = Gnocca;
result_trained_model.new_model_equation.test_results.Gnocca.test_predictions = lx_test_gnocca.LxNewEquationPred;
result_trained_model.new_model_equation.test_results.Gnocca.metrics = results_test_gnocca("LxNewEquation",["RMSE","R2","Corr Coeff"]);

% compute test performance on Po Tolle
Tolle = struct();
results_test_tolle = compute_metrics(lx_test_tolle.LxObs,lx_test_tolle.LxNewEquationPred,algorithm_names(4), results_test_tolle);
result_trained_model.new_model_equation.test_results.Tolle = Tolle;
result_trained_model.new_model_equation.test_results.Tolle.test_predictions = lx_test_tolle.LxNewEquationPred;
result_trained_model.new_model_equation.test_results.Tolle.metrics = results_test_tolle("LxNewEquation",["RMSE","R2","Corr Coeff"]);

% compute test performance on Po Dritta
Dritta = struct();
results_test_dritta = compute_metrics(lx_test_dritta.LxObs,lx_test_dritta.LxNewEquationPred,algorithm_names(4), results_test_dritta);
result_trained_model.new_model_equation.test_results.Dritta = Dritta;
result_trained_model.new_model_equation.test_results.Dritta.test_predictions = lx_test_dritta.LxNewEquationPred;
result_trained_model.new_model_equation.test_results.Dritta.metrics = results_test_dritta("LxNewEquation",["RMSE","R2","Corr Coeff"]);


%% Store prediction in dataset
lx_training_dataset.RF_Pred = result_trained_model.random_forest.validation_results.validation_predictions;
lx_training_dataset.Lsb_Pred = result_trained_model.lsboost.validation_results.validation_predictions;

lx_test_gnocca.RF_Pred = result_trained_model.random_forest.test_results.Gnocca.test_predictions;
lx_test_gnocca.Lsb_Pred = result_trained_model.lsboost.test_results.Gnocca.test_predictions;

lx_test_tolle.RF_Pred = result_trained_model.random_forest.test_results.Tolle.test_predictions;
lx_test_tolle.Lsb_Pred = result_trained_model.lsboost.test_results.Tolle.test_predictions;

lx_test_dritta.RF_Pred = result_trained_model.random_forest.test_results.Dritta.test_predictions;
lx_test_dritta.Lsb_Pred = result_trained_model.lsboost.test_results.Dritta.test_predictions;

%%
% save test results on Po Gnocca-Tolle-Dritta
Gnocca_Tolle_Dritta = struct();
result_trained_model.random_forest.test_results.Gnocca_Tolle_Dritta = Gnocca_Tolle_Dritta;
result_trained_model.random_forest.test_results.Gnocca_Tolle_Dritta.test_predictions = [lx_test_gnocca.RF_Pred;lx_test_tolle.RF_Pred;lx_test_dritta.RF_Pred];
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.random_forest.test_results.Gnocca_Tolle_Dritta.test_predictions, algorithm_names(1), results_test);
result_trained_model.random_forest.test_results.Gnocca_Tolle_Dritta.metrics = results_test("random_forest",["RMSE","R2","Corr Coeff"]);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.random_forest.test_results.Gnocca_Tolle_Dritta.test_predictions,pwbTable,algorithm_names(1),pwbX);

% save test results on Po Gnocca-Tolle-Dritta
Gnocca_Tolle_Dritta = struct();
result_trained_model.lsboost.test_results.Gnocca_Tolle_Dritta = Gnocca_Tolle_Dritta;
result_trained_model.lsboost.test_results.Gnocca_Tolle_Dritta.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(lx_test_dataset, {'DateObs','LxOldEquationPred', 'LxNewEquationPred', 'Branch', 'DatasetType'}));
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), result_trained_model.lsboost.test_results.Gnocca_Tolle_Dritta.test_predictions, algorithm_names(2), results_test);
result_trained_model.lsboost.test_results.Gnocca_Tolle_Dritta.metrics = results_test("lsboost",["RMSE","R2","Corr Coeff"]);
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), result_trained_model.lsboost.test_results.Gnocca_Tolle_Dritta.test_predictions,pwbTable,algorithm_names(2),pwbX);

% save test results on Po Gnocca-Tolle-Dritta
Gnocca_Tolle_Dritta = struct();
result_trained_model.old_model_equation.test_results.Gnocca_Tolle_Dritta = Gnocca_Tolle_Dritta;
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), lx_test_dataset.LxOldEquationPred, algorithm_names(3), results_test);
result_trained_model.old_model_equation.test_results.Gnocca_Tolle_Dritta.metrics = results_test("LxOldEquation",["RMSE","R2","Corr Coeff"]);
result_trained_model.old_model_equation.test_results.Gnocca_Tolle_Dritta.test_predictions = lx_test_dataset.LxOldEquationPred;
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), lx_test_dataset.LxOldEquationPred,pwbTable,algorithm_names(3),pwbX);

% save test results on Po Gnocca-Tolle-Dritta
Gnocca_Tolle_Dritta = struct();
result_trained_model.new_model_equation.test_results.Gnocca_Tolle_Dritta = Gnocca_Tolle_Dritta;
results_test = compute_metrics(lx_test_dataset(:,targetFeatureName), lx_test_dataset.LxNewEquationPred, algorithm_names(4), results_test);
result_trained_model.new_model_equation.test_results.Gnocca_Tolle_Dritta.metrics = results_test("LxNewEquation",["RMSE","R2","Corr Coeff"]);
result_trained_model.new_model_equation.test_results.Gnocca_Tolle_Dritta.test_predictions = lx_test_dataset.LxNewEquationPred;
pwbTable = create_pwb_table(lx_test_dataset(:, targetFeatureName), lx_test_dataset.LxNewEquationPred,pwbTable,algorithm_names(4),pwbX);

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