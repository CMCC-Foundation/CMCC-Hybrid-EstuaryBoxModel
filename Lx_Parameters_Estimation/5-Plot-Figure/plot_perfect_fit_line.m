addpath(genpath("..\0-Dataset"));
addpath(genpath("..\3_Trained-Models"));
load("..\0-Dataset\LX_OBS_WITH_FEATURES.mat");
load("..\3-Trained-Models\Trained-model-k-4.mat");

subplot(1,2,1);
plotPerfectFit(lx_dataset.Lx_OBS, result_trained_model.random_forest.predictions, 'Random forest');

subplot(1,2,2);
plotPerfectFit(lx_dataset.Lx_OBS, result_trained_model.lsboost.predictions, 'Lsboost');

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
