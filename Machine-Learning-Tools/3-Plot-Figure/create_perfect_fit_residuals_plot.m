function [] = create_perfect_fit_residuals_plot(resumePredictionsTable, algorithm_names, response, titleSubplot)
    f = figure;
    f.Position = [0 0 1920 1000];
    
    for i = 1:numel(algorithm_names)
        subplot(2,numel(algorithm_names),i);
        plotPerfectFit(resumePredictionsTable(:,1), resumePredictionsTable(:,i+1), algorithm_names(i));
    
        subplot(2,numel(algorithm_names),i+numel(algorithm_names));
        resumeTable = createResumeTable(resumePredictionsTable(:,1), resumePredictionsTable(:,i+1), response);
        plotResidualBar(resumeTable, algorithm_names(i), response); 
    end
    sgtitle(titleSubplot);
end

function [] = plotPerfectFit(obs, pred, modelName)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end
    hAx=gca;                  
    plot(obs,pred, '.','MarkerSize',18, ...
        'MarkerFaceColor',[0.00,0.45,0.74],'MarkerEdgeColor','auto');
    hold on;
    xy = linspace(0, 30,30 );
    plot(xy,xy,'k-','LineWidth',1.3);
    hAx.LineWidth=1;
    xlim([0 30]);
    ylim([0 30]);
    xlabel('True response');
    ylabel('Predicted response');
    title(modelName);
    legend('Observations','Perfect prediction','Location','northwest');
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
    ylim([0 30]);
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