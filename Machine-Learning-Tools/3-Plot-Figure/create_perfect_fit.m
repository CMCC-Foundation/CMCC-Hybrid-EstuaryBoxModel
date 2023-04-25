function [] = create_perfect_fit(resumePredictions,algorithm_names,addBoundPerfectFit, percentageBoundPerfectFit)
    f = figure;
    f.Position = [0 0 1500 450];
    %f.Position = [0 0 1000 1000];

    for i = 1:numel(algorithm_names)
        %subplot(2,numel(algorithm_names),i);
        subplot(1,3,i);
        plotPerfectFit( ...
            resumePredictions(:,1), ...
            resumePredictions(:,i+1), ...
            algorithm_names(i), ...
            addBoundPerfectFit, ...
            percentageBoundPerfectFit);
    end
end

function [] = plotPerfectFit(obs, pred, modelName, addBound, percentageBound)
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
    xy = linspace(0, 40, 40);
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
    xlim([0 40]);
    ylim([0 40]);
    xlabel('True response');
    ylabel('Predicted response');
    title(modelName);
    legend(legendName,'Location','northwest');
    set(gca,'FontSize',14);
    grid on;
    hold off;
end