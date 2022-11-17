addpath(genpath('..\..\0-Dataset\training_test_2016_2019'));
addpath(genpath('..\..\1_Trained-Models\training_test_2016_2019'));
addpath(genpath('..\..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\..\0-Dataset\training_test_2016_2019\Salinity-Training-Dataset.mat");
load("..\..\0-Dataset\training_test_2016_2019\Salinity-Test-Dataset.mat");
load("..\..\1-Trained-Models\training_test_2016_2019\Salinity-Trained-Tested-model-k-10.mat");

algorithm_names = {'random forest', 'lsboost', 'neural network'};
response = 'Salinity_Obs';

%% Training dataset
training_table_results = array2table([salinity_training_dataset.Salinity_Obs ...
    result_trained_model.random_forest.validation_results.validation_predictions...
    result_trained_model.lsboost.validation_results.validation_predictions...
    result_trained_model.neural_network.validation_results.validation_predictions ...
],"VariableNames",{'real_sal', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, "Training dataset");
compare_real_pred_obs(training_table_results, algorithm_names, "Training dataset", "Salinity (psu)");

%% Test dataset
test_table_results = array2table([salinity_test_dataset.Salinity_Obs ...
    result_trained_model.random_forest.test_results.test_predictions...
    result_trained_model.lsboost.test_results.test_predictions...
    result_trained_model.neural_network.test_results.test_predictions ...
],"VariableNames",{'real_sal', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset");
compare_real_pred_obs(test_table_results, algorithm_names, "Test dataset", "Salinity (psu)");
