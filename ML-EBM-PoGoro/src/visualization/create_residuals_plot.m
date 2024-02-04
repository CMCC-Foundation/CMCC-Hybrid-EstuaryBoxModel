function [f] = create_residuals_plot(resumePredictions,algorithm_names, response)
%CREATE_RESIDUALS_PLOT This function plot a response plot with residuals
%   Input:
%   1) resumePredictions - table with a summary of observed and predicted
%   values
%   2) algorithm_names - string array with the names of the trained models
%   3) response - the name of the target value

    f = figure;
    f.Position = [0 -5 1068 1001];
    tiledlayout(2,2);

    for i = 1:numel(algorithm_names)
        nexttile
        resumeTable = createResumeTable( ...
            resumePredictions(:,1), ...
            resumePredictions(:,i+1), ...
            response);

        plotResidualBar( ...
            resumeTable, ...
            algorithm_names(i), ...
            response);
    end
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
    ylim([0 40]);
    yticks([0 5 10 15 20 25 30 35 40]);
    yticklabels({'0','5','10','15','20','25','30','35','40'});
    legend('True','Predicted','Errors','Location','northwest');
    xlabel('Record number');
    ylabel('Response (psu)');
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