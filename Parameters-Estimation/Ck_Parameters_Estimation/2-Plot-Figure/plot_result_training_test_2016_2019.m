addpath(genpath('..\0-Dataset\'));
addpath(genpath('..\1_Trained-Models\'));
load("..\0-Dataset\Ck-obs-by-rf\ck_train_dataset_rf.mat");
load("..\0-Dataset\Ck-obs-by-rf\ck_test_dataset_rf.mat");
load("..\0-Dataset\Ck-obs-by-lsboost\ck_train_dataset_lsboost.mat");
load("..\0-Dataset\Ck-obs-by-lsboost\ck_test_dataset_lsboost.mat");
load("..\1-Trained-Models\Ck-Obs-By-RF\ck_model_rf.mat");
load("..\1-Trained-Models\Ck-Obs-By-LSBoost\ck-model-lsboost.mat");

algorithm_names = {'RF','LSBoost'};
response = 'CkObs';

%% Training dataset
training_table_results = array2table([ ...
    ck_train_dataset_rf.CkObs ...
    ck_model_rf.validation_results.validation_predictions...
    ck_train_dataset_lsboost.CkObs ...
    ck_model_lsboost.validation_results.validation_predictions...
],"VariableNames",{'real_ck_rf' ,'rf_pred', 'real_ck_lsb','lsb_pred'});

create_perfect_fit(training_table_results,algorithm_names,true,30);
create_residuals_plot(training_table_results,algorithm_names,response);

%% Test dataset
test_table_results = array2table([ ...
    ck_test_dataset_rf.CkObs ...
    ck_model_rf.test_results.test_predictions...
    ck_test_dataset_lsboost.CkObs ...
    ck_model_lsboost.test_results.test_predictions...
],"VariableNames",{'real_ck_rf' ,'rf_pred', 'real_ck_lsb','lsb_pred'});

create_perfect_fit(test_table_results,algorithm_names,true,30);
create_residuals_plot(test_table_results,algorithm_names,response);

%% Perfect fit function
function [] = create_perfect_fit(resumePredictions,algorithm_names,addBoundPerfectFit, percentageBoundPerfectFit)
    f = figure;
    f.Position = [0 0 1140 450];
    j = 1;
    for i = 1:2:width(resumePredictions)
        subplot(1,2,j);
        plotPerfectFit( ...
            resumePredictions(:,i), ...
            resumePredictions(:,i+1), ...
            algorithm_names(j), ...
            addBoundPerfectFit, ...
            percentageBoundPerfectFit);
        j=j+1;
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
    xy = linspace(0, 1100, 1100);
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
    xlim([0 1100]);
    ylim([0 1100]);
    xlabel('True response');
    ylabel('Predicted response');
    title(modelName);
    legend(legendName,'Location','northwest');
    set(gca,'FontSize',14);
    grid on;
    hold off;
end

%% Residuals plot function
function [] = create_residuals_plot(resumePredictions,algorithm_names, response)
    f = figure;
    f.Position = [0 0 1140 450];

    j = 1;
    for i = 1:2:width(resumePredictions)
        subplot(1,2,j);
        resumeTable = createResumeTable( ...
            resumePredictions(:,i), ...
            resumePredictions(:,i+1), ...
            response);

        plotResidualBar( ...
            resumeTable, ...
            algorithm_names(j), ...
            response);
        j=j+1;
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
    ylim([0 1100]);
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
