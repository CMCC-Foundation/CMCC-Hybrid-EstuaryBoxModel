function [] = create_perfect_fit_residuals_plot(resumePredictionsTable, algorithm_names, response, titleSubplot, addBoundPerfectFit, percentageBoundPerfectFit)
    f = figure;
    f.Position = [0 0 1920 1000];
    
    for i = 1:numel(algorithm_names)
        subplot(2,numel(algorithm_names),i);
        plotPerfectFit(resumePredictionsTable(:,1), resumePredictionsTable(:,i+1), algorithm_names(i), addBoundPerfectFit, percentageBoundPerfectFit);
    
        subplot(2,numel(algorithm_names),i+numel(algorithm_names));
        resumeTable = createResumeTable(resumePredictionsTable(:,1), resumePredictionsTable(:,i+1), response);
        plotResidualBar(resumeTable, algorithm_names(i), response); 
    end
    sgtitle(titleSubplot);
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
    plot(obs,pred, '.','MarkerSize',18, ...
        'MarkerFaceColor',[0.00,0.45,0.74],'MarkerEdgeColor','auto');
    hold on;
    xy = linspace(0, 630, 630);
    plot(xy,xy,'k-','LineWidth',1.3);
    if(addBound)
        xyUpperBound = xy + percentageBound*xy/100;
        xyLowerBound = xy - percentageBound*xy/100;
        plot(xy,xyUpperBound, 'r--', 'LineWidth',1.3);
        plot(xy,xyLowerBound, 'r--', 'LineWidth',1.3);
        legendName = {"Observations","Perfect prediction", ...
            strcat(string(percentageBound), "% of deviation")};
    end
    hAx.LineWidth=1;
    xlim([0 630]);
    ylim([0 630]);
    xlabel('True response');
    ylabel('Predicted response');
    title(modelName);
    legend(legendName,'Location','northwest');
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
    hAx = gca;
    index = linspace(1, height(resumeTable), height(resumeTable));
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
    
    hAx.LineWidth=1;
    xlim([0 max(index)+1]);
    ylim([0 630]);
    legend('True','Predicted','Errors','Location','northwest');
    xlabel('Record number');
    ylabel('Response');
    title(modelName);
    set(gca,'FontSize',12);
    grid on;
    box on;
    hold off;
end

function [resumeTable] = createResumeTable(obs, pred, response)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end
    resumeTable = array2table([obs pred abs(obs - pred)],...
        'VariableNames',{response,'Predicted','Residuals'} );
    resumeTable = sortrows(resumeTable,response,{'ascend'});
    resumeTable.ID = linspace(1, numel(obs), numel(obs))';
end