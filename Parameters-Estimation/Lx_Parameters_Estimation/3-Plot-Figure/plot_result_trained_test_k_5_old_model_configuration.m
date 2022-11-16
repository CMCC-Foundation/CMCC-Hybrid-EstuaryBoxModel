addpath(genpath('..\0-Dataset\'));
addpath(genpath('..\1-Trained-Models\Trained-Test-Results-k-5-old-model-configuration\'));
addpath(genpath('..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\0-Dataset\LX_OBS_WITH_FEATURES.mat");
load("..\1-Trained-Models\Trained-Test-Results-k-5-old-model-configuration\Trained-Tested-model-k-5.mat");

algorithm_names = {'old model','random forest', 'lsboost'};
response = 'Lx_Obs';

%% Training dataset
training_table_results = array2table([lx_dataset.Lx_OBS(lx_dataset.Dataset_Type == 'CALIBRATION RANGE') ...
    lx_dataset.Lx_Model(lx_dataset.Dataset_Type == 'CALIBRATION RANGE')...
    result_trained_model.random_forest.validation_results.validation_predictions ...
    result_trained_model.lsboost.validation_results.validation_predictions...
],"VariableNames",{'real_lx','old_model_pred', 'rf_pred', 'lsb_pred'});

create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, "Training dataset");
compare_real_pred_obs(training_table_results, algorithm_names, "Training dataset", "Km");

%% Test dataset
test_table_results = array2table([lx_dataset.Lx_OBS(lx_dataset.Dataset_Type == 'VALIDATION RANGE') ...
    lx_dataset.Lx_Model(lx_dataset.Dataset_Type == 'VALIDATION RANGE')...
    result_trained_model.random_forest.test_results.test_predictions...
    result_trained_model.lsboost.test_results.test_predictions...
],"VariableNames",{'real_lx', 'old_model_pred', 'rf_pred', 'lsb_pred'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, "Test dataset");
compare_real_pred_obs(test_table_results, algorithm_names, "Test dataset", "Km");