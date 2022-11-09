addpath(genpath('..\0-Dataset\'));
addpath(genpath('..\1-Trained-Models\Trained-All-Dataset-k-5\'));
addpath(genpath('..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\0-Dataset\LX_OBS_WITH_FEATURES.mat");
load("..\1-Trained-Models\Trained-All-Dataset-k-5\Trained-model-k-5.mat");

algorithm_names = {'random forest', 'lsboost'};
response = 'Lx_OBS';

%% Training dataset
training_table_results = array2table([lx_dataset.Lx_OBS ...
    result_trained_model.random_forest.validation_results.validation_predictions...
    result_trained_model.lsboost.validation_results.validation_predictions...
],"VariableNames",{'real_lx', 'rf_pred', 'lsb_pred'});

create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, "Training dataset");
compare_real_pred_obs(training_table_results, algorithm_names, "Training dataset", "Km");