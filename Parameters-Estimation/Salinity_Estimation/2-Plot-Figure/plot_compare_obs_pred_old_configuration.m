addpath(genpath('..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model'));
addpath(genpath('..\1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model'));
load("..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Training-Dataset_2016_2017.mat");
load("..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Test-Dataset_2018_2019.mat");
load("..\1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Trained-Tested-model-k-10-old-configuration.mat");

algorithm_names = {'old model','random forest', 'lsboost', 'neural network'};
response = 'Salinity_Obs';

%% Training
training_response_obs = table2array(salinity_training_dataset(:,response));
training_response_old_model = table2array(salinity_training_dataset(:,"Salinity_Old_model"));

f = figure;
f.Position = [0 0 1920 1000];

subplot(4,1,1);
plotPerfectFit(training_response_obs, training_response_old_model , algorithm_names(1));

subplot(4,1,2);
plotPerfectFit(training_response_obs, result_trained_model.random_forest.validation_results.validation_predictions , algorithm_names(2));

subplot(4,1,3);
plotPerfectFit(training_response_obs, result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(3));

subplot(4,1,4);
plotPerfectFit(training_response_obs, result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(4));

sgtitle('Training results on 2016-2017 observations');

%% Test 2018
test_response_obs = table2array(salinity_test_dataset_2018_2019(salinity_test_dataset_2018_2019.Year == 2018,response));
test_response_old_model = table2array(salinity_test_dataset_2018_2019(salinity_test_dataset_2018_2019.Year == 2018, "Salinity_Old_model"));

f = figure;
f.Position = [0 0 1920 1000];

subplot(4,1,1);
plotPerfectFit(test_response_obs, test_response_old_model, algorithm_names(1));

subplot(4,1,2);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions, algorithm_names(2));

subplot(4,1,3);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions, algorithm_names(3));

subplot(4,1,4);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions, algorithm_names(4));

sgtitle('Test results on 2018 observations');

%% Test 2019
test_response_obs = table2array(salinity_test_dataset_2018_2019(salinity_test_dataset_2018_2019.Year == 2019,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(3,1,1);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions, algorithm_names(2));

subplot(3,1,2);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions, algorithm_names(3));

subplot(3,1,3);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions, algorithm_names(4));

sgtitle('Test results on 2019 observations');

%% Test 2018 - 2019
test_response_obs = table2array(salinity_test_dataset_2018_2019(:,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(3,1,1);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(2));

subplot(3,1,2);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(3));

subplot(3,1,3);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(4));

sgtitle('Test results on 2018-2019 observations');



function [] = plotPerfectFit(obs, pred, modelName)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    
    end
    
    x = linspace(1,numel(obs),numel(obs));
    
    plot(x,obs, '-','LineWidth',1.3);
    hold on;
    plot(x,pred,'-','LineWidth',1.3);
    xlim([0 max(x)+1]);
    ylim([0 30]);
    xlabel('Record number');
    ylabel('Salinity (psu)');
    title(modelName);
    legend('Observed','Modelled','Location','northeast');
    set(gca,'FontSize',12);
    grid on;
    hold off;
end