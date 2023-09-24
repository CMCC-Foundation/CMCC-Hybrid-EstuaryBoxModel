function [] = plot_residuals_bar(resumeTable,modelName, response, yylim)
    obs = resumeTable(:,response);
    pred = resumeTable.Predicted;
    
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end
    hAx = gca;
    index = linspace(1, height(resumeTable), height(resumeTable));
    hold on;
    plot(index, obs, '.','LineWidth',0.5, 'Color',[0.00,0.00,1.00], ...
        'MarkerSize',20, 'MarkerEdgeColor','auto');
    plot(index, pred, '.','LineWidth',0.5, 'Color',[0.93,0.69,0.13], ...
        'MarkerSize',20, 'MarkerEdgeColor','auto');
    
    for i = 1 : numel(index)
        plot([index(i), index(i)], [obs(i), pred(i)], ...
            'Color', [1.00,0.00,0.00], 'LineWidth', 1,  ...
            'MarkerSize',6, 'MarkerEdgeColor','auto');
    end
    
    hAx.LineWidth=1.4;
    xlim([-2 max(index)+2]);
    ylim([0 yylim]);
    legend('True','Predicted','Errors','Location','northwest');
    xlabel('Record number');
    ylabel('Response');
    title(modelName);
    set(gca,'FontSize',14);
    grid on;
    box on;
    hold off;
end