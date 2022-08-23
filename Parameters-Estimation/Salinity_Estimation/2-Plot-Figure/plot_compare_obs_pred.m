addpath(genpath('..\0-Dataset'));
addpath(genpath('..\1_Trained-Models'));
load("..\0-Dataset\Salinity-Training-Dataset.mat");
load("..\0-Dataset\Salinity-Test-Dataset.mat");
load("..\1-Trained-Models\Salinity-Trained-Tested-model-k-10.mat");

algorithm_names = {'random forest', 'lsboost', 'neural network'};
response = 'Salinity_Obs';

%% Training
training_response_obs = table2array(salinity_training_dataset(:,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(3,1,1);
plotPerfectFit(training_response_obs, result_trained_model.random_forest.predictions, algorithm_names(1));

subplot(3,1,2);
plotPerfectFit(training_response_obs, result_trained_model.lsboost.predictions, algorithm_names(2));

subplot(3,1,3);
plotPerfectFit(training_response_obs, result_trained_model.neural_network.predictions, algorithm_names(3));

sgtitle('Training results');

%% Test
test_response_obs = table2array(salinity_test_dataset(:,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(3,1,1);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test.predictions, algorithm_names(1));

subplot(3,1,2);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test.predictions, algorithm_names(2));

subplot(3,1,3);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test.predictions, algorithm_names(3));

sgtitle('Test results');

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