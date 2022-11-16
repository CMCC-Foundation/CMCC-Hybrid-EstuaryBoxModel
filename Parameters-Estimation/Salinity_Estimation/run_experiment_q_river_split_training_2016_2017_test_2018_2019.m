%% The following script group all the experiment carried out in this paper:
%  Given the dataset, we trained different machine learning and deep learning 
%  algorithm to perform a regression task. 
%  We also use hyperparameters optimization and cross-validation
%  with k = 5. 
%  We use data from 2016-2017 to train and validate our model, and data 
%  from 2018-2019 examples to test it.
%  The aim is to compare the old model performance with these new models.

%% Add to path subdirectory
addpath(genpath('0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted'));

%% Set import dataset settings
filepath = "0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted\SALINITY_Q_RIVER_CLASS_SPLIT.xlsx";
nVars = 9;
dataRange = "A2:I1462";
sheetName = "Salinity_Q_river_splitted";
varNames = ["Year","Qriver", "Qll", "Qtide", "Sll", "Socean", "SalinityObs", "SalinityOldmodel", "QriverClass"]; 
varTypes = ["int16", "double", "double", "double", "double","double","double", "double", "categorical"];

[salinity_dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes);

%% Remove observations with missing values
salinity_dataset = remove_missing_data_features(salinity_dataset);

%% Plot bar with sample distribution with respect the year and the q_river_class
plot_bar_q_river_splitted_obs_by_year(salinity_dataset);
plot_bar_q_river_splitted_obs_by_training_test_year(salinity_dataset);

%% Plot correlation matrix of all features and response in dataset
plot_corrplot(removevars(salinity_dataset, {'Year', 'SalinityOldmodel', 'QriverClass'}));

%% User select which class of q_river must be used into the experiment
q_river_selected_class = 0;
fprintf("Running salinity estimation experiment ...\n")
fprintf("=================================================================================\n")
while 1
    fprintf("Select one of the following configuration for 'Q_river_class': ");
    fprintf("\n1) LOW");
    fprintf("\n2) STRONG");
    fprintf("\n---------------------------------------------------------------------------------");
    q_river_selected_class = input("\nEnter: ");
    
    if(isnumeric(q_river_selected_class))
        if(q_river_selected_class == 1)
            q_river_selected_class="LOW";
            break;
        elseif(q_river_selected_class == 2)
            q_river_selected_class="STRONG";
            break;
        else
            fprintf("Invalid configuration selected!");
            fprintf("\n---------------------------------------------------------------------------------\n");
        end
    end
end

fprintf(strcat("Running salinity estimation experiment using ", q_river_selected_class," Q_river_class\n"));
fprintf("---------------------------------------------------------------------------------\n");

%% Split salinity dataset with q_river_class in training and test set
salinity_training_dataset = salinity_dataset((salinity_dataset.Year == 2016 | salinity_dataset.Year == 2017) & strcmp(string(salinity_dataset.QriverClass), q_river_selected_class),:);
salinity_test_dataset_2018 = salinity_dataset((salinity_dataset.Year == 2018) & strcmp(string(salinity_dataset.QriverClass), q_river_selected_class),:);
salinity_test_dataset_2019 = salinity_dataset((salinity_dataset.Year == 2019) & strcmp(string(salinity_dataset.QriverClass), q_river_selected_class),:);
salinity_test_dataset_2018_2019 = salinity_dataset((salinity_dataset.Year == 2018 | salinity_dataset.Year == 2019) & strcmp(string(salinity_dataset.QriverClass), q_river_selected_class),:);

save("0-Dataset/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/SALINITY_Q_RIVER_CLASS_SPLIT.mat", "salinity_dataset");
save(strcat("0-Dataset/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/",q_river_selected_class,"/Salinity-Training-Dataset_2016_2017.mat"), "salinity_training_dataset");
save(strcat("0-Dataset/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/",q_river_selected_class,"/Salinity-Test-Dataset_2018_2019.mat"), "salinity_test_dataset_2018_2019");

%% Plot boxplot
plot_boxplot(removevars(salinity_training_dataset,{'Year', 'SalinityOldmodel', 'QriverClass'}),...
    removevars(salinity_test_dataset_2018,{'Year', 'SalinityOldmodel', 'QriverClass'}),...
    removevars(salinity_test_dataset_2019,{'Year', 'SalinityOldmodel', 'QriverClass'}),...
    removevars(salinity_test_dataset_2018_2019,{'Year', 'SalinityOldmodel', 'QriverClass'}));

%% Create table for k-fold cross validation results
algorithm_names = {'random_forest', 'lsboost', 'neural_network', 'old_model' };

results_training = table('Size', [4 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_2018_dataset = table('Size', [4 8], ...
    'VariableTypes', {'double', 'double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_2019_dataset = table('Size', [4 8], ...
    'VariableTypes', {'double', 'double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test_2018_2019_dataset = table('Size', [4 8], ...
    'VariableTypes', {'double', 'double', 'double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

result_trained_model = struct();

%% Set target feature for the machine and deep learning model
targetFeatureName = 'SalinityObs';

%% Set maxObjectiveEvaluations as maximum number of objective functions to
%  be evaluated in the optimization process
max_objective_evaluations = 45;

%% Set k to be use in k-fold cross validation
k = 5;

%% Training random forest model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.random_forest = random_forest_function(removevars(salinity_training_dataset, {'Year', 'SalinityOldmodel', 'QriverClass'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1), results_training);
result_trained_model.random_forest.validation_results.metrics = results_training("random_forest",:);

% save test results
test_2018_dataset = struct();
test_2019_dataset = struct();
test_2018_2019_dataset = struct();
result_trained_model.random_forest.test_results.test_2018_dataset = test_2018_dataset;
result_trained_model.random_forest.test_results.test_2019_dataset = test_2019_dataset;
result_trained_model.random_forest.test_results.test_2018_2019_dataset = test_2018_2019_dataset;

% test only on 2018 observations
result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(salinity_test_dataset_2018, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2018_dataset = compute_metrics(salinity_test_dataset_2018(:, targetFeatureName), result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions, algorithm_names(1), results_test_2018_dataset);
result_trained_model.random_forest.test_results.test_2018_dataset.metrics = results_test_2018_dataset("random_forest",:);

% test only on 2019 observations 
result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(salinity_test_dataset_2019, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2019_dataset = compute_metrics(salinity_test_dataset_2019(:, targetFeatureName), result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions, algorithm_names(1), results_test_2019_dataset);
result_trained_model.random_forest.test_results.test_2019_dataset.metrics = results_test_2019_dataset("random_forest",:);

% test on 2018-2019 observations
result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions = result_trained_model.random_forest.model.predictFcn(removevars(salinity_test_dataset_2018_2019, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2018_2019_dataset = compute_metrics(salinity_test_dataset_2018_2019(:, targetFeatureName), result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(1), results_test_2018_2019_dataset);
result_trained_model.random_forest.test_results.test_2018_2019_dataset.metrics = results_test_2018_2019_dataset("random_forest",:);


%% Training lsboost model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(2), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.lsboost = lsboost_function(removevars(salinity_training_dataset, {'Year', 'SalinityOldmodel', 'QriverClass'}),targetFeatureName,max_objective_evaluations, k);
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2), results_training);
result_trained_model.lsboost.validation_results.metrics = results_training("lsboost",:);

% save test results
test_2018_dataset = struct();
test_2019_dataset = struct();
test_2018_2019_dataset = struct();
result_trained_model.lsboost.test_results.test_2018_dataset = test_2018_dataset;
result_trained_model.lsboost.test_results.test_2019_dataset = test_2019_dataset;
result_trained_model.lsboost.test_results.test_2018_2019_dataset = test_2018_2019_dataset;

% test only on 2018 observations
result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(salinity_test_dataset_2018, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2018_dataset = compute_metrics(salinity_test_dataset_2018(:, targetFeatureName), result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions, algorithm_names(2), results_test_2018_dataset);
result_trained_model.lsboost.test_results.test_2018_dataset.metrics = results_test_2018_dataset("lsboost",:);

% test only on 2019 observations 
result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(salinity_test_dataset_2019, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2019_dataset = compute_metrics(salinity_test_dataset_2019(:, targetFeatureName), result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions, algorithm_names(2), results_test_2019_dataset);
result_trained_model.lsboost.test_results.test_2019_dataset.metrics = results_test_2019_dataset("lsboost",:);

% test on 2018-2019 observations
result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions = result_trained_model.lsboost.model.predictFcn(removevars(salinity_test_dataset_2018_2019, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2018_2019_dataset = compute_metrics(salinity_test_dataset_2018_2019(:, targetFeatureName), result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(2), results_test_2018_2019_dataset);
result_trained_model.lsboost.test_results.test_2018_2019_dataset.metrics = results_test_2018_2019_dataset("lsboost",:);


%% Training neural network model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(3), " with k=", string(k), "\n"));
fprintf("===================================================================\n");

% save training results and performance
result_trained_model.neural_network = neural_network_function(removevars(salinity_training_dataset, {'Year', 'SalinityOldmodel', 'QriverClass'}),targetFeatureName, 1, 5, 10, 100,max_objective_evaluations, k);
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(3), results_training);
result_trained_model.neural_network.validation_results.metrics = results_training("neural_network",:);

% save test results
test_2018_dataset = struct();
test_2019_dataset = struct();
test_2018_2019_dataset = struct();
result_trained_model.neural_network.test_results.test_2018_dataset = test_2018_dataset;
result_trained_model.neural_network.test_results.test_2019_dataset = test_2019_dataset;
result_trained_model.neural_network.test_results.test_2018_2019_dataset = test_2018_2019_dataset;

% test only on 2018 observations
result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(salinity_test_dataset_2018, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2018_dataset = compute_metrics(salinity_test_dataset_2018(:, targetFeatureName), result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions, algorithm_names(3), results_test_2018_dataset);
result_trained_model.neural_network.test_results.test_2018_dataset.metrics = results_test_2018_dataset("neural_network",:);

% test only on 2019 observations 
result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(salinity_test_dataset_2019, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2019_dataset = compute_metrics(salinity_test_dataset_2019(:, targetFeatureName), result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions, algorithm_names(3), results_test_2019_dataset);
result_trained_model.neural_network.test_results.test_2019_dataset.metrics = results_test_2019_dataset("neural_network",:);

% test on 2018-2019 observations
result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions = result_trained_model.neural_network.model.predictFcn(removevars(salinity_test_dataset_2018_2019, {'Year','SalinityObs','SalinityOldmodel', 'QriverClass'}));
results_test_2018_2019_dataset = compute_metrics(salinity_test_dataset_2018_2019(:, targetFeatureName), result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(3), results_test_2018_2019_dataset);
result_trained_model.neural_network.test_results.test_2018_2019_dataset.metrics = results_test_2018_2019_dataset("neural_network",:);

%% Update metrics from old model
results_training = compute_metrics(salinity_training_dataset(:, targetFeatureName), salinity_training_dataset(:,"SalinityOldmodel"), algorithm_names(4), results_training);
results_test_2018_dataset = compute_metrics(salinity_test_dataset_2018(:, targetFeatureName), salinity_test_dataset_2018(:,"SalinityOldmodel"), algorithm_names(4), results_test_2018_dataset);
results_test_2019_dataset = compute_metrics(salinity_test_dataset_2019(:, targetFeatureName), salinity_test_dataset_2019(:,"SalinityOldmodel"), algorithm_names(4), results_test_2019_dataset);
results_test_2018_2019_dataset = compute_metrics(salinity_test_dataset_2018_2019(:, targetFeatureName), salinity_test_dataset_2018_2019(:,"SalinityOldmodel"), algorithm_names(4), results_test_2018_2019_dataset);

%% Save results
writetable(results_training, strcat("1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/", q_river_selected_class,"/Results-salinity-training.xlsx"), 'WriteRowNames',true);
writetable(results_test_2018_dataset, strcat("1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/",q_river_selected_class,"/Results-salinity-test-2018-model.xlsx"), 'WriteRowNames',true);
writetable(results_test_2019_dataset, strcat("1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/",q_river_selected_class,"/Results-salinity-test-2019-model.xlsx"), 'WriteRowNames',true);
writetable(results_test_2018_2019_dataset, strcat("1-Trained-Models/training_2016_2017_test_2018_2019_comparing_old_model/Q_river_Class_splitted/",q_river_selected_class,"/Results-salinity-test-2018-2019-model.xlsx"), 'WriteRowNames',true);
save(strcat("1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted\",q_river_selected_class,"\Salinity-Trained-Tested-model.mat"),"result_trained_model");