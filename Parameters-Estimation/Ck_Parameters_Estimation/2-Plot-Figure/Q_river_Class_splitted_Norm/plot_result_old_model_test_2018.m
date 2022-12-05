%% User select which class of q_river must be used into the experiment
q_river_selected_class = 0;
fprintf("Running ck estimation plot experiment result ...\n")
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

fprintf(strcat("Plotting ck estimation experiment result using ", q_river_selected_class," Q_river_class\n"));
fprintf("---------------------------------------------------------------------------------\n");

%% Add path
addpath(genpath('..\..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted_Norm\'));
addpath(genpath(strcat('..\..\1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted_Norm\',q_river_selected_class)));
addpath(genpath('..\..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load('..\..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted_Norm\CK_Q_RIVER_CLASS_SPLIT_NORM.mat');
load(strcat("..\..\1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Q_river_Class_splitted_Norm\",q_river_selected_class,'\Ck-Trained-Tested-model-norm.mat'));

algorithm_names = {'old model','random forest', 'lsboost', 'neural network'};
response = 'CkObs';

%% Training dataset
training_table_results = array2table([ck_dataset.CkObs((ck_dataset.Year == 2016 | ck_dataset.Year == 2017) & strcmp(string(ck_dataset.QriverClass), q_river_selected_class)) ...
    ck_dataset.CkOldmodel((ck_dataset.Year == 2016 | ck_dataset.Year == 2017) & strcmp(string(ck_dataset.QriverClass), q_river_selected_class))...
    result_trained_model.random_forest.validation_results.validation_predictions...
    result_trained_model.lsboost.validation_results.validation_predictions...
    result_trained_model.neural_network.validation_results.validation_predictions ...
],"VariableNames",{'real_sal','old_model_pred', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(training_table_results, algorithm_names, response, strcat("Training dataset 2016 - 2017 with ",q_river_selected_class, " Qriver"));

%% Test dataset
test_table_results = array2table([ck_dataset.CkObs((ck_dataset.Year == 2018) & strcmp(string(ck_dataset.QriverClass), q_river_selected_class))...
    ck_dataset.CkOldmodel((ck_dataset.Year == 2018) & strcmp(string(ck_dataset.QriverClass), q_river_selected_class))...
    result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions...
    result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions...
    result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions ...
],"VariableNames",{'real_sal', 'old_model_pred', 'rf_pred', 'lsb_pred', 'nn_pred'});

create_perfect_fit_residuals_plot(test_table_results, algorithm_names, response, strcat("Test 2018 with ",q_river_selected_class, " Qriver"),false,0);