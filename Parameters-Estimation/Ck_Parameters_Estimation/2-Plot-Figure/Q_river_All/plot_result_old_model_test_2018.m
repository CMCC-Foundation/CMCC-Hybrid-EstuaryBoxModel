addpath(genpath('..\..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All'));
addpath(genpath('..\..\1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All'));
addpath(genpath('..\..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All\CK_OLD_MODEL_PREDICTIONS.mat")
load("..\..\1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All\Ck-Trained-Tested-model-k-5-old-configuration.mat");

algorithm_names = {'EBM','Random Forest', 'Lsboost', 'Neural Network'};
response = 'CK_Obs';

%% Training dataset
training_table_results = array2table([ck_dataset.CK_Obs((ck_dataset.Year == 2016 | ck_dataset.Year == 2017)) ...
    ck_dataset.Ck_old_model(ck_dataset.Year == 2016 | ck_dataset.Year == 2017)...
    result_trained_model.random_forest.validation_results.validation_predictions...
    result_trained_model.lsboost.validation_results.validation_predictions...
    result_trained_model.neural_network.validation_results.validation_predictions ...
],"VariableNames",{'real_ck','old_model_pred', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, "Training dataset 2016 - 2017", true,30);
%compare_real_pred_obs(training_table_results, algorithm_names, "Training dataset 2016 - 2017", "ck");

%% Test dataset
test_table_results = array2table([ck_dataset.CK_Obs(ck_dataset.Year == 2018)...
    ck_dataset.Ck_old_model(ck_dataset.Year == 2018)...
    result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions...
    result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions...
    result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions ...
],"VariableNames",{'real_ck', 'old_model_pred', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset 2018", true,30);
%compare_real_pred_obs(test_table_results, algorithm_names, "Test dataset 2018", "ck");

