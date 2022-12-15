addpath(genpath('..\..\0-Dataset\training_test_2016_2019_new_cal'));
addpath(genpath('..\..\1_Trained-Models\training_test_2016_2019_new_cal'));
addpath(genpath('..\..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\..\0-Dataset\training_test_2016_2019_new_cal\Salinity-Training-Dataset-new-cal.mat");
load("..\..\0-Dataset\training_test_2016_2019_new_cal\Salinity-Test-Dataset-new-cal.mat");
load("..\..\1-Trained-Models\training_test_2016_2019_new_cal\Salinity-Trained-Tested-model-new-cal.mat");

algorithm_names = {'EBM','random forest', 'lsboost', 'neural network'};
response = 'SalinityObs';

%% Training dataset
training_table_results = array2table([salinity_training_dataset.SalinityObs ...
    table2array(result_trained_model.EBM.validation_results.validation_predictions)...
    result_trained_model.random_forest.validation_results.validation_predictions...
    result_trained_model.lsboost.validation_results.validation_predictions...
    result_trained_model.neural_network.validation_results.validation_predictions ...
],"VariableNames",{'ebm_pred','real_sal', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, "Training dataset",true,30);

%% Test dataset
test_table_results = array2table([salinity_test_dataset.SalinityObs ...
    table2array(result_trained_model.EBM.test_results.test_predictions)...
    result_trained_model.random_forest.test_results.test_predictions...
    result_trained_model.lsboost.test_results.test_predictions...
    result_trained_model.neural_network.test_results.test_predictions ...
],"VariableNames",{'ebm_pred','real_sal', 'rf_pred', 'lsb_pred', 'nn_pred'});


create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset",true,30);