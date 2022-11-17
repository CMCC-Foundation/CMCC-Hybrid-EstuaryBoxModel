addpath(genpath('..\..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All'));
addpath(genpath('..\..\1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All'));
addpath(genpath('..\..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All\CK_OLD_MODEL_PREDICTIONS.mat")
load("..\..\1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_All\Ck-Trained-Tested-model-k-5-old-configuration.mat");

algorithm_names = {'old model','random forest', 'lsboost', 'neural network'};
response = 'CK_Obs';

%% Test dataset
test_table_results = array2table([ck_dataset.CK_Obs(ck_dataset.Year == 2019)...
    ck_dataset.Ck_old_model(ck_dataset.Year == 2019)...
    result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions...
    result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions...
    result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions ...
],"VariableNames",{'real_ck', 'old_model_pred', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset 2019");
compare_real_pred_obs(test_table_results, algorithm_names, "Test dataset 2019", "ck");