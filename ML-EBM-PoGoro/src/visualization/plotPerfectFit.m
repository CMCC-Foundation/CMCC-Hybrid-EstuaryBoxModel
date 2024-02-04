function [] = plotPerfectFit(obs, pred, modelName, addBound, percentageBound)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end
    hAx=gca;                  
    legendName = {'Observations','Perfect prediction'};
    plot(obs, pred, 'o','MarkerSize', 5, 'MarkerFaceColor', [0.00,0.45,0.74], ...
        'Color', [0.00,0.00,1.00]);
    hold on;
    xy = linspace(0, 31, 31);
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
    xlim([0 31]);
    ylim([0 31]);
    xticks([0 5 10 15 20 25 30 ]);
    xticklabels({'0','5','10','15','20','25','30'});
    yticks([0 5 10 15 20 25 30 ]);
    yticklabels({'0','5','10','15','20','25','30'});
    xlabel('Observed salinity (psu)');
    ylabel('Predicted salinity (psu)');
    title(modelName);
    %legend(legendName,'Location','northwest');
    set(gca,'FontSize',14);
    grid on;
    hold off;
end