addpath(genpath('..\0-Dataset\training_test_2016_2019'));
addpath(genpath('..\1_Trained-Models\training_test_2016_2019'));
load("..\0-Dataset\training_test_2016_2019\Salinity-Training-Dataset.mat");
load("..\0-Dataset\training_test_2016_2019\Salinity-Test-Dataset.mat");
load("..\1-Trained-Models\training_test_2016_2019\Salinity-Trained-Tested-model-k-10.mat");

algorithm_names = {'random forest', 'lsboost', 'neural network'};
response = 'Salinity_Obs';

training_response_obs = table2array(salinity_training_dataset(:,response));

%% Training results
f = figure;
f.Position = [0 0 1920 1000];

subplot(2,3,1);
plotPerfectFit(training_response_obs, result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(1));

subplot(2,3,2);
plotPerfectFit(training_response_obs, result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(2));

subplot(2,3,3);
plotPerfectFit(training_response_obs, result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(3));

subplot(2,3,4);
resumeTable = createResumeTable(training_response_obs, result_trained_model.random_forest.validation_results.validation_predictions, response);
plotResidualBar(resumeTable, algorithm_names(1), response);

subplot(2,3,5);
resumeTable = createResumeTable(training_response_obs, result_trained_model.lsboost.validation_results.validation_predictions, response);
plotResidualBar(resumeTable, algorithm_names(2), response);

subplot(2,3,6);
resumeTable = createResumeTable(training_response_obs, result_trained_model.neural_network.validation_results.validation_predictions, response);
plotResidualBar(resumeTable, algorithm_names(3), response);

sgtitle('Training results');

%% Test results
test_response_obs = table2array(salinity_test_dataset(:,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(2,3,1);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_predictions, algorithm_names(1));

subplot(2,3,2);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_predictions, algorithm_names(2));

subplot(2,3,3);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_predictions, algorithm_names(3));

subplot(2,3,4);
resumeTable = createResumeTable(test_response_obs, result_trained_model.random_forest.test_results.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(1), response);

subplot(2,3,5);
resumeTable = createResumeTable(test_response_obs, result_trained_model.lsboost.test_results.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(2), response);

subplot(2,3,6);
resumeTable = createResumeTable(test_response_obs, result_trained_model.neural_network.test_results.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(3), response);

sgtitle('Test results');

function [] = plotPerfectFit(obs, pred, modelName)
if (istable(obs))
    obs = table2array(obs);
end

if(istable(pred))
    pred = table2array(pred);

end

plot(obs,pred, '.','MarkerSize',18, ...
    'MarkerFaceColor',[0.00,0.45,0.74],'MarkerEdgeColor','auto');
hold on;
xy = linspace(0, 30,30 );
plot(xy,xy,'k-','LineWidth',1.3);
xlim([0 30]);
ylim([0 30]);
xlabel('True response');
ylabel('Predicted response');
title(modelName);
legend('Observations','Perfect prediction','Location','northwest');
set(gca,'FontSize',12);
grid on;
hold off;
end

function [] = plotResidualBar(resumeTable,modelName, response)
obs = resumeTable(:,response);
pred = resumeTable.Predicted;

if (istable(obs))
    obs = table2array(obs);
end

if(istable(pred))
    pred = table2array(pred);

end

index = linspace(0, height(resumeTable), height(resumeTable));

hold on;
plot(index, obs, '.','LineWidth',0.5, 'Color',[0.00,0.45,0.74], ...
    'MarkerSize',18, 'MarkerEdgeColor','auto');
plot(index, pred, '.','LineWidth',0.5, 'Color',[0.93,0.69,0.13], ...
    'MarkerSize',18, 'MarkerEdgeColor','auto');

for i = 1 : numel(index)
    plot([index(i), index(i)], [obs(i), pred(i)], ...
        'Color', [0.85,0.33,0.10], 'LineWidth', 1,  ...
        'MarkerSize',6, 'MarkerEdgeColor','auto');
end

xlim([0 max(index)+5]);
ylim([0 30]);
legend('True','Predicted','Errors','Location','northwest');
xlabel('Record number');
ylabel('Response');
title(modelName);
set(gca,'FontSize',12);
grid on;
box on;
hold off;
end

function [resumeTable] = createResumeTable(obs, pred, response)
resumeTable = array2table([obs pred abs(obs - pred)],...
    'VariableNames',{response,'Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,response,{'ascend'});
resumeTable.ID = linspace(1, numel(obs), numel(obs))';

end