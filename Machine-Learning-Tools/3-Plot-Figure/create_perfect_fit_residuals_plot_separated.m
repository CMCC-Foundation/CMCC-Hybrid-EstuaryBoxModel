function [] = create_perfect_fit_residuals_plot_separated(resumePredictionsTableTraining, resumePredictionsTableTest, algorithm_names, response, titleSubplot, addBoundPerfectFit, percentageBoundPerfectFit)
    f = figure;
    %f.Position = [0 0 1920 100];
    f.Position = [0 0 1000 1000];

    for i = 1:numel(algorithm_names)
        %subplot(1,numel(algorithm_names),i);
        subplot(2,2,i);

        %plotPerfectFit(resumePredictionsTableTraining(:,1), resumePredictionsTableTraining(:,i+1), algorithm_names(i), addBoundPerfectFit, percentageBoundPerfectFit);
        resumeTableTraining = createResumeTable(resumePredictionsTableTraining(:,1), resumePredictionsTableTraining(:,i+1), response);
        plotResidualBar(resumeTableTraining, algorithm_names(i), response);
        
        %subplot(2,numel(algorithm_names),i+numel(algorithm_names));
        %plotPerfectFit(resumePredictionsTableTest(:,1), resumePredictionsTableTest(:,i+1), algorithm_names(i), addBoundPerfectFit, percentageBoundPerfectFit);
        %resumeTableTest = createResumeTable(resumePredictionsTableTest(:,1), resumePredictionsTableTest(:,i+1), response);
        %plotResidualBar(resumeTableTest, algorithm_names(i), response); 
       
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
    xy = linspace(0, 650, 650);
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
    xlim([0 650]);
    ylim([0 650]);
    xlabel('True response');
    ylabel('Predicted response');
    title(modelName);
    legend(legendName,'Location','northwest');
    set(gca,'FontSize',14);
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
    ylim([0 650]);
    legend('True','Predicted','Errors','Location','northwest');
    xlabel('Record number');
    ylabel('Response');
    title(modelName);
    set(gca,'FontSize',14);
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