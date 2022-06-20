subplot(1,3,1);
plotPerfectFit(training_dataset.Lx_OBS, result_trained_model.random_forest.predictions, 'Random forest');

subplot(1,3,2);
plotPerfectFit(training_dataset.Lx_OBS, result_trained_model.lsboost.predictions, 'Lsboost');

subplot(1,3,3);
plotPerfectFit(training_dataset.Lx_OBS, result_trained_model.neural_network.predictions, 'Neural network');

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
