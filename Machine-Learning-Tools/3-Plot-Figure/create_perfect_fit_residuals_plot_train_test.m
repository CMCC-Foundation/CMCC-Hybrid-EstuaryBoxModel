function [] = create_perfect_fit_residuals_plot_train_test(resumePredictionsTableTraining, resumePredictionsTableTest, algorithm_names, response, addBoundPerfectFit, percentageBoundPerfectFit)
    f = figure;
    f.Position = [0 0 1920 500];
    
    for i = 1:numel(algorithm_names)
        %subplot(1,numel(algorithm_names),i);
        subplot(2,2,i);
        %{
        plotPerfectFit(resumePredictionsTableTraining(:,1), resumePredictionsTableTraining(:,i+1), ...
            resumePredictionsTableTest(:,1), resumePredictionsTableTest(:,i+1),...
            algorithm_names(i), addBoundPerfectFit, percentageBoundPerfectFit);
        %}
        resumeTable = createResumeTable(resumePredictionsTableTraining(:,1), resumePredictionsTableTraining(:,i+1), ...
            resumePredictionsTableTest(:,1), resumePredictionsTableTest(:,i+1), response);
        plotResidualBar(resumeTable, algorithm_names(i),response);
    end
end

function [] = plotPerfectFit(obs_train, pred_train, obs_test,pred_test, modelName, addBound, percentageBound)
    if (istable(obs_train))
        obs_train = table2array(obs_train);
    end
    
    if(istable(pred_train))
        pred_train = table2array(pred_train);
    end

    if (istable(obs_test))
        obs_test = table2array(obs_test);
    end
    
    if(istable(pred_test))
        pred_test = table2array(pred_test);
    end
    hAx=gca;                  
    legendName = {'Training observations','Testing observations','Perfect prediction'};
    plot(obs_train,pred_train, '.','MarkerSize',20, ...
        'MarkerFaceColor',[0.00 0.00 1.00],'MarkerEdgeColor','auto');
    hold on;
    plot(obs_test,pred_test, '.','MarkerSize',20, ...
        'MarkerFaceColor',[1.00 0.00, 0.00],'MarkerEdgeColor','auto');

    xy = linspace(0, 30, 30);
    plot(xy,xy,'-','LineWidth',1.7,'Color',[0 0 0]);
    if(addBound)
        xyUpperBound = xy + percentageBound*xy/100;
        xyLowerBound = xy - percentageBound*xy/100;
        plot(xy,xyUpperBound, '--', 'LineWidth',1.7,'Color',[0 0 0]);
        plot(xy,xyLowerBound, '--', 'LineWidth',1.7, 'Color',[0 0 0]);
        legendName = {'Training observations','Testing observations',...
            'Perfect prediction',strcat(string(percentageBound), "% of deviation")};
    end
    hAx.LineWidth=1.4;
    xlim([0 30]);
    ylim([0 30]);
    xlabel('True response (Km)');
    ylabel('Predicted response (Km)');
    title(modelName);
    legend(legendName,'Location','northwest');
    set(gca,'FontSize',14);
    grid on;
    hold off;
end

function [] = plotResidualBar(resumeTable,modelName, response)
    
    %% Train
    obs_train = resumeTable(resumeTable.Type == "train",response);
    pred_train = resumeTable.Predicted(resumeTable.Type=="train");
    index_train = resumeTable.ID(resumeTable.Type=="train");
    
    if (istable(obs_train))
        obs_train = table2array(obs_train);
    end
    
    if(istable(pred_train))
        pred_train = table2array(pred_train);
    end
    hAx = gca;
    
    hold on;
    plot(index_train, obs_train, '.','LineWidth',0.5, 'Color',[0.00,0.00,1.00], ...
        'MarkerSize',20, 'MarkerEdgeColor','auto');
    plot(index_train, pred_train, '.','LineWidth',0.5, 'Color',[0.93,0.69,0.13], ...
        'MarkerSize',20, 'MarkerEdgeColor','auto');
    
    for i = 1 : numel(index_train)
        plot([index_train(i), index_train(i)], [obs_train(i), pred_train(i)], ...
            'Color', [0.85,0.33,0.10], 'LineWidth', 1,  ...
            'MarkerSize',6, 'MarkerEdgeColor','auto');
    end

    %% Test
    obs_test = resumeTable(resumeTable.Type == "test",response);
    pred_test = resumeTable.Predicted(resumeTable.Type=="test");
    index_test = resumeTable.ID(resumeTable.Type=="test");
    
    if (istable(obs_test))
        obs_test = table2array(obs_test);
    end
    
    if(istable(pred_test))
        pred_test = table2array(pred_test);
    end
    hAx = gca;
    
    hold on;
    plot(index_test, obs_test, '.','LineWidth',0.5, 'Color',[0.00,0,0], ...
        'MarkerSize',20, 'MarkerEdgeColor','auto');
    plot(index_test, pred_test, '.','LineWidth',0.5, 'Color',[0.47,0.67,0.19], ...
        'MarkerSize',20, 'MarkerEdgeColor','auto');
    
    for i = 1 : numel(index_test)
        plot([index_test(i), index_test(i)], [obs_test(i), pred_test(i)], ...
            'Color', [0.85,0.33,0.10], 'LineWidth', 1,  ...
            'MarkerSize',6, 'MarkerEdgeColor','auto');
    end
    
    hAx.LineWidth=1.4;
    xlim([0 max(index_train)+1]);
    ylim([0 30]);
    legend('True','Predicted','Errors','Location','northwest');
    xlabel('Record number');
    ylabel('Response');
    title(modelName);
    set(gca,'FontSize',14);
    grid on;
    box on;
    hold off;
end

function [resumeTable] = createResumeTable(obs_train, pred_train,obs_test,pred_test,response)
    if (istable(obs_train))
        obs_train = table2array(obs_train);
    end
    
    if(istable(pred_train))
        pred_train = table2array(pred_train);
    end

    if (istable(obs_test))
        obs_test = table2array(obs_test);
    end
    
    if(istable(pred_test))
        pred_test = table2array(pred_test);
    end        	 	 	

    resumeTable_train = array2table([obs_train pred_train abs(obs_train - pred_train)],...
        'VariableNames',{response,'Predicted','Residuals'} );
    resumeTable_train.Type = string(repmat('train',numel(obs_train), 1));

    resumeTable_test = array2table([obs_test pred_test abs(obs_test - pred_test)],...
        'VariableNames',{response,'Predicted','Residuals'} );
    resumeTable_test.Type = string(repmat('test',numel(obs_test), 1));
    resumeTable = [resumeTable_train; resumeTable_test];

    resumeTable = sortrows(resumeTable,response,{'ascend'});

    resumeTable.ID = linspace(1, height(resumeTable), height(resumeTable))';
end