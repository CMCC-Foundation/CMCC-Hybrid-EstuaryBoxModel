function [] = plot_perfect_fit(obs, pred, modelName, addBound, percentageBound, xylim)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end
    hAx=gca;
    legendName = {'Observations','Perfect prediction'};
    plot(obs,pred, '.','MarkerSize',20, ...
        'MarkerFaceColor',[0.00,0.00,0.1],'MarkerEdgeColor','auto');
    hold on;
    xy = linspace(0, xylim, xylim);
    plot(xy,xy,'k-','LineWidth',2);
    if(addBound)
        xyUpperBound = xy + percentageBound*xy/100;
        xyLowerBound = xy - percentageBound*xy/100;
        plot(xy,xyUpperBound, 'r--', 'LineWidth',2);
        plot(xy,xyLowerBound, 'r--', 'LineWidth',2);
        legendName = {"Observations","Perfect prediction", ...
            strcat(string(percentageBound), "% of deviation")};
    end
    hAx.LineWidth=1.4;
    xlim([0 xylim]);
    ylim([0 xylim]);
    xlabel('True response');
    ylabel('Predicted response');
    title(modelName);
    legend(legendName,'Location','northwest');
    set(gca,'FontSize',14);
    grid on;
    hold off;
end