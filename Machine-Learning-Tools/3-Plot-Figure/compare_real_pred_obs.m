function [] = compare_real_pred_obs(resumePredictionsTable, algorithm_names, titleSubplot, ylab)
    f = figure;
    f.Position = [0 0 1920 1000];
    
    for i = 1:numel(algorithm_names)
        subplot(numel(algorithm_names),1,i);
        plotPerfectFit(resumePredictionsTable(:,1), resumePredictionsTable(:,i+1), algorithm_names(i), ylab);
    end
    sgtitle(titleSubplot);
end

function [] = plotPerfectFit(obs, pred, modelName, ylab)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end
    
    hAx = gca;
    x = linspace(1,numel(obs),numel(obs));
    plot(x,obs, '-','LineWidth',1.3,'MarkerFaceColor','#0072BD');
    hold on;
    plot(x,pred,'-','LineWidth',1.3,'MarkerFaceColor','#D95319');
    xlim([0 max(x)+1]);
    ylim([0 630]);
    xlabel('Record number');
    ylabel(ylab);
    title(modelName);
    hAx.LineWidth=1;
    legend('Observed','Modelled','Location','northeast');
    set(gca,'FontSize',12);
    grid on;
    hold off;
end