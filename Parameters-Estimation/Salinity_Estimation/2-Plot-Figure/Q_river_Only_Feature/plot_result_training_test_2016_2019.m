addpath(genpath('..\..\0-Dataset\training_test_2016_2019'));
addpath(genpath('..\..\1_Trained-Models\training_only_qriver'));
addpath(genpath('..\..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\..\0-Dataset\training_test_2016_2019\Salinity-Training-Dataset.mat");
load("..\..\0-Dataset\training_test_2016_2019\Salinity-Test-Dataset.mat");
load("..\..\1-Trained-Models\training_only_qriver\Salinity-Trained-Tested-model.mat");

algorithm_names = {'exp1'};
response = 'SalinityObs';

%% Training dataset
training_table_results = array2table([salinity_training_dataset.SalinityObs ...
    result_trained_model.exp1.validation_results.validation_predictions...
],"VariableNames",{'RealObs','exp1'});
create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, "Training dataset",true,30);

%% Test dataset
test_table_results = array2table([salinity_test_dataset.SalinityObs ...
    result_trained_model.exp1.test_results.test_predictions...
],"VariableNames",{'RealObs','exp1'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset",true,30);