numElDataset = numel(training_dataset.Lx_OBS);
resumeTable = array2table([training_dataset.Lx_OBS result_trained_model.random_forest.predictions...
    abs(training_dataset.Lx_OBS - result_trained_model.random_forest.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(1,3,1);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Random forest');

resumeTable = array2table([training_dataset.Lx_OBS result_trained_model.lsboost.predictions...
    abs(training_dataset.Lx_OBS - result_trained_model.lsboost.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(1,3,2);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Lsboost');

resumeTable = array2table([training_dataset.Lx_OBS result_trained_model.neural_network.predictions...
    abs(training_dataset.Lx_OBS - result_trained_model.neural_network.predictions)],...
    'VariableNames',{'Lx_OBS','Predicted','Residuals'} );
resumeTable = sortrows(resumeTable,{'Lx_OBS'},{'ascend'});
resumeTable.ID = linspace(1,numElDataset,numElDataset)';
subplot(1,3,3);
plotResidualBar(resumeTable.ID, resumeTable.Lx_OBS,resumeTable.Predicted, 'Neural network');


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