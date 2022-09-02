addpath(genpath('..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model'));
addpath(genpath('..\1_Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model'));
load("..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Training-Dataset_2016_2017.mat");
load("..\0-Dataset\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Test-Dataset_2018_2019.mat");
load("..\1-Trained-Models\training_2016_2017_test_2018_2019_comparing_old_model\Salinity-Trained-Tested-model-k-10-old-configuration.mat");

algorithm_names = {'old model','random forest', 'lsboost', 'neural network'};
response = 'Salinity_Obs';

training_response_obs = table2array(salinity_training_dataset(:,response));
training_response_old_model = table2array(salinity_training_dataset(:,"Salinity_Old_model"));

%% Training results
f = figure;
f.Position = [0 0 1920 1000];

subplot(2,4,1);
plotPerfectFit(training_response_obs, training_response_old_model , algorithm_names(1))

subplot(2,4,2);
plotPerfectFit(training_response_obs, result_trained_model.random_forest.validation_results.validation_predictions, algorithm_names(2));

subplot(2,4,3);
plotPerfectFit(training_response_obs, result_trained_model.lsboost.validation_results.validation_predictions, algorithm_names(3));

subplot(2,4,4);
plotPerfectFit(training_response_obs, result_trained_model.neural_network.validation_results.validation_predictions, algorithm_names(4));

subplot(2,4,5);
resumeTable = createResumeTable(training_response_obs, training_response_old_model, response);
plotResidualBar(resumeTable, algorithm_names(1), response);

subplot(2,4,6);
resumeTable = createResumeTable(training_response_obs, result_trained_model.random_forest.validation_results.validation_predictions, response);
plotResidualBar(resumeTable, algorithm_names(2), response);

subplot(2,4,7);
resumeTable = createResumeTable(training_response_obs, result_trained_model.lsboost.validation_results.validation_predictions, response);
plotResidualBar(resumeTable, algorithm_names(3), response);

subplot(2,4,8);
resumeTable = createResumeTable(training_response_obs, result_trained_model.neural_network.validation_results.validation_predictions, response);
plotResidualBar(resumeTable, algorithm_names(4), response);

sgtitle('Training results on 2016-2017 observations');

%% Test results 2018
test_response_obs = table2array(salinity_test_dataset_2018_2019(salinity_test_dataset_2018_2019.Year == 2018,response));
test_response_old_model = table2array(salinity_test_dataset_2018_2019(salinity_test_dataset_2018_2019.Year == 2018, "Salinity_Old_model"));

f = figure;
f.Position = [0 0 1920 1000];

subplot(2,4,1);
plotPerfectFit(test_response_obs, test_response_old_model , algorithm_names(1))

subplot(2,4,2);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions, algorithm_names(2));

subplot(2,4,3);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions, algorithm_names(3));

subplot(2,4,4);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions, algorithm_names(4));

subplot(2,4,5);
resumeTable = createResumeTable(test_response_obs, test_response_old_model, response);
plotResidualBar(resumeTable, algorithm_names(1), response);

subplot(2,4,6);
resumeTable = createResumeTable(test_response_obs, result_trained_model.random_forest.test_results.test_2018_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(2), response);

subplot(2,4,7);
resumeTable = createResumeTable(test_response_obs, result_trained_model.lsboost.test_results.test_2018_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(3), response);

subplot(2,4,8);
resumeTable = createResumeTable(test_response_obs, result_trained_model.neural_network.test_results.test_2018_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(4), response);

sgtitle('Test results on 2018 observations');


%% Test results 2019
test_response_obs = table2array(salinity_test_dataset_2018_2019(salinity_test_dataset_2018_2019.Year == 2019,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(2,3,1);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions, algorithm_names(2));

subplot(2,3,2);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions, algorithm_names(3));

subplot(2,3,3);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions, algorithm_names(4));

subplot(2,3,4);
resumeTable = createResumeTable(test_response_obs, result_trained_model.random_forest.test_results.test_2019_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(2), response);

subplot(2,3,5);
resumeTable = createResumeTable(test_response_obs, result_trained_model.lsboost.test_results.test_2019_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(3), response);

subplot(2,3,6);
resumeTable = createResumeTable(test_response_obs, result_trained_model.neural_network.test_results.test_2019_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(4), response);

sgtitle('Test results on 2019 observations');

%% Test results 2018-2019
test_response_obs = table2array(salinity_test_dataset_2018_2019(:,response));

f = figure;
f.Position = [0 0 1920 1000];

subplot(2,3,1);
plotPerfectFit(test_response_obs, result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(2));

subplot(2,3,2);
plotPerfectFit(test_response_obs, result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(3));

subplot(2,3,3);
plotPerfectFit(test_response_obs, result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions, algorithm_names(4));

subplot(2,3,4);
resumeTable = createResumeTable(test_response_obs, result_trained_model.random_forest.test_results.test_2018_2019_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(2), response);

subplot(2,3,5);
resumeTable = createResumeTable(test_response_obs, result_trained_model.lsboost.test_results.test_2018_2019_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(3), response);

subplot(2,3,6);
resumeTable = createResumeTable(test_response_obs, result_trained_model.neural_network.test_results.test_2018_2019_dataset.test_predictions, response);
plotResidualBar(resumeTable, algorithm_names(4), response);

sgtitle('Test results on 2018-2019 observations');

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

%{
function [] = plotRealObs(obs)
if (istable(obs))
    obs = table2array(obs);
end
obs = sort(obs);
plot(obs, '.','LineWidth',0.5, 'Color',[0.00,0.45,0.74], ...
    'MarkerSize',18, 'MarkerEdgeColor','auto');

xlim([0 max(numel(obs))+5]);
ylim([0 50]);
legend('Observations','Location','northwest');
xlabel('Record number');
ylabel('Observation');
title('Observations distribution');
grid on;
end
%}

function [resumeTable] = createResumeTable(obs, pred, response)
resumeTable = array2table([obs pred abs(obs - pred)],...
    'VariableNames',{response,'Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,response,{'ascend'});
resumeTable.ID = linspace(1, numel(obs), numel(obs))';

end