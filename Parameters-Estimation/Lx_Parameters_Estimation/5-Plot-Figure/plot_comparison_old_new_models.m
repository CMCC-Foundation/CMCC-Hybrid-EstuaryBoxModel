addpath(genpath("..\0-Dataset"));
addpath(genpath("..\1_Trained-Models"));
load("..\0-Dataset\Old_Model_Training_Test_Prediction.mat");
load("..\1-Trained-Models\Trained-Tested-model-k-4.mat");

%% Plot Training Results comparing old and new models
lx_training = Old_Model_Training.LX_Obs;
old_model_lx_pred_training = Old_Model_Training.LX_Pred;

% Perfect line prediction plot
subplot(2,3,1);
plotPerfectFit(lx_training, old_model_lx_pred_training, 'Old model');

subplot(2,3,2);
plotPerfectFit(lx_training, result_trained_tested_model.random_forest.predictions, 'Random forest');

subplot(2,3,3);
plotPerfectFit(lx_training, result_trained_tested_model.lsboost.predictions, 'Lsboost');

%{
subplot(2,4,4);
plotPerfectFit(lx_training, result_trained_tested_model.neural_network.predictions, 'Neural network');
%}

% Residuals plot
numElDataset = numel(lx_training);
resumeTable = array2table([lx_training old_model_lx_pred_training...
    abs(lx_training - old_model_lx_pred_training)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,3,4);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Old model');

numElDataset = numel(lx_training);
resumeTable = array2table([lx_training result_trained_tested_model.random_forest.predictions...
    abs(lx_training - result_trained_tested_model.random_forest.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,3,5);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Random forest');

resumeTable = array2table([lx_training result_trained_tested_model.lsboost.predictions...
    abs(lx_training - result_trained_tested_model.lsboost.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,3,6);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Lsboost');

%{
resumeTable = array2table([lx_training result_trained_tested_model.neural_network.predictions...
    abs(lx_training - result_trained_tested_model.neural_network.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,4,8);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Neural network');
%}

sgtitle('Training results with old and new models');

%% Plot Training Results comparing old and new models
lx_test = Old_Model_Test.LX_Obs;
old_model_lx_pred_test = Old_Model_Test.LX_Pred;

figure;

% Perfect line prediction plot

subplot(2,3,1);
plotPerfectFit(lx_test, old_model_lx_pred_test, 'Old model');

subplot(2,3,2);
plotPerfectFit(lx_test, result_trained_tested_model.random_forest.test.predictions, 'Random forest');

subplot(2,3,3);
plotPerfectFit(lx_test, result_trained_tested_model.lsboost.test.predictions, 'Lsboost');

%{
subplot(2,4,4);
plotPerfectFit(lx_test, result_trained_tested_model.neural_network.test.predictions, 'Neural network');
%}

% Residuals plot
numElDataset = numel(lx_test);
resumeTable = array2table([lx_test old_model_lx_pred_test...
    abs(lx_test - old_model_lx_pred_test)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,3,4);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Old model');

numElDataset = numel(lx_test);
resumeTable = array2table([lx_test result_trained_tested_model.random_forest.test.predictions...
    abs(lx_test - result_trained_tested_model.random_forest.test.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,3,5);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Random forest');

resumeTable = array2table([lx_test result_trained_tested_model.lsboost.test.predictions...
    abs(lx_test - result_trained_tested_model.lsboost.test.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,3,6);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Lsboost');

%{
resumeTable = array2table([lx_test result_trained_tested_model.neural_network.test.predictions...
    abs(lx_test - result_trained_tested_model.neural_network.test.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(2,4,8);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Neural network');
%}

sgtitle('Testing results with old and new models');


%% function to plot perfect line prediction plot 
function [] = plotPerfectFit(obs, pred, modelName)
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
    grid on;
    hold off;
end

%% Function to plot residuals prediction plot
function [] = plotResidualBar(index,vo,vp,modelName)
    % vo: values observed
    % vp: values predicted
    hold on;
    plot(index, vo, '.','LineWidth',0.5, 'Color',[0.00,0.45,0.74], ...
        'MarkerSize',18, 'MarkerEdgeColor','auto');
    plot(index, vp, '.','LineWidth',0.5, 'Color',[0.93,0.69,0.13], ...
        'MarkerSize',18, 'MarkerEdgeColor','auto');
    
    for i = 1 : numel(index)
        plot([index(i), index(i)], [vo(i), vp(i)], ...
            'Color', [0.85,0.33,0.10], 'LineWidth', 1,  ...
            'MarkerSize',6, 'MarkerEdgeColor','auto');
    end
    
    xlim([0 max(index)+2]);
    ylim([0 max(vo)+5]);
    legend('True','Predicted','Errors','Location','northwest');
    xlabel('Record number');
    ylabel('Response');
    title(modelName);
    grid on;
    hold off;
end