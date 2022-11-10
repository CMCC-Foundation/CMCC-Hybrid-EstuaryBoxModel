addpath(genpath('..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model'));
addpath(genpath('..\1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model'));
addpath(genpath('..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\SALINITY_OLD_MODEL_PREDICTIONS.mat")
load("..\1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Trained-Tested-model-k-5-old-configuration.mat");

algorithm_names = {'old model','random forest', 'lsboost', 'neural network'};
response = 'Salinity_Obs';

%% Test dataset
test_table_results = array2table([salinity_dataset.Salinity_Obs(salinity_dataset.Year == 2018 | salinity_dataset.Year == 2019 )...
    salinity_dataset.Salinity_Old_model(salinity_dataset.Year == 2018 | salinity_dataset.Year == 2019)...
    result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions...
    result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions...
    result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions ...
],"VariableNames",{'real_sal', 'old_model_pred', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset 2018 - 2019");
compare_real_pred_obs(test_table_results, algorithm_names, "Test dataset 2018 - 2019", "Salinity (psu)");

