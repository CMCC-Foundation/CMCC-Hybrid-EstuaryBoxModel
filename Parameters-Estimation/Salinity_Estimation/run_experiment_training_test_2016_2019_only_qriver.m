%% The following script group all the experiment carried out in this paper:
%  We use 80% of examples to train and validate our model, and 20% examples to test it. 

%% Add to path subdirectory
addpath(genpath('0-Dataset\training_test_2016_2019'));
addpath(genpath('..\..\Machine-Learning-Tools\1-Utility'));
addpath(genpath('..\..\Machine-Learning-Tools\2-Machine-Learning-Function'));
addpath(genpath('..\..\Machine-Learning-Tools\3-Plot-Figure'));
addpath(genpath('1_Trained-Models\training_test_2016_2019'));

load("0-Dataset\training_test_2016_2019\Salinity-Training-Dataset.mat");
load("0-Dataset\training_test_2016_2019\Salinity-Test-Dataset.mat");

xTrain = salinity_training_dataset.Qriver;
yTrain = salinity_training_dataset.SalinityObs;
xTest = salinity_test_dataset.Qriver;
yTest = salinity_test_dataset.SalinityObs;

%% Create table validation results
algorithm_names = {'exp1'};
pwbX = [1 5 10 20 30];
pwbXRowNames = string();

for i = 1:numel(pwbX)
    pwbXRowNames(i) = strcat('PWB', num2str(pwbX(i)));
end

results_test = table('Size', [numel(algorithm_names) 8], ...
    'VariableTypes', {'double','double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE','NRMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

pwbTable = table('Size',[numel(pwbX) numel(algorithm_names)],...
    'VariableTypes', repmat({'double'}, 1, numel(algorithm_names)), ...
    'VariableNames', algorithm_names,...
    'RowNames', pwbXRowNames);

result_trained_model = struct();

%% Training exp1 model
fprintf("\n===================================================================\n");
fprintf(strcat("Training model using ", algorithm_names(1), "\n"));
fprintf("===================================================================\n");

% save training results and performance
[result_trained_model.exp1] = exponential_fit_function(xTrain,yTrain, xTest,yTest);

% save test results
results_test= compute_metrics(yTest, result_trained_model.exp1.test_results.test_predictions, algorithm_names(1), results_test);
result_trained_model.exp1.test_results.metrics = results_test("exp1",:);
pwbTable = create_pwb_table(yTest, result_trained_model.exp1.test_results.test_predictions,pwbTable,algorithm_names(1),pwbX);

clc;
disp(results_test);
disp(pwbTable);

writetable(results_test, '1-Trained-Models\training_only_qriver\/Results-salinity-test-model.xlsx', 'WriteRowNames',true);
writetable(pwbTable, "1-Trained-Models\training_only_qriver\/pwbTable.xlsx", "WriteRowNames", true);
save("1-Trained-Models\training_only_qriver\Salinity-Trained-Tested-model.mat","result_trained_model");