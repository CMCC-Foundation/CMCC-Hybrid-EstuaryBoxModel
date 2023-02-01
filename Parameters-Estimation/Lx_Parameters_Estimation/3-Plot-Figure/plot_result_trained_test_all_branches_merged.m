addpath(genpath('..\0-Dataset\Po-all-branches\all-branches-merged'));
addpath(genpath('..\..\..\Machine-Learning-Tools\3-Plot-Figure'));
load("..\0-Dataset\Po-all-branches\all-branches-merged\lx_training_dataset.mat");
load("..\0-Dataset\Po-all-branches\all-branches-merged\lx_test_dataset.mat");

algorithm_names = {'random forest', 'lsboost'};
response = 'LxObs';

goro_train = array2table([lx_training_dataset.LxObs(lx_training_dataset.Branch == "PO GORO",:), ...
    lx_training_dataset.RandomForest_Prediction(lx_training_dataset.Branch == "PO GORO",:),...
    lx_training_dataset.Lsboost_Prediction(lx_training_dataset.Branch == "PO GORO",:)],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

goro_test = array2table([lx_test_dataset.LxObs(lx_test_dataset.Branch == "PO GORO",:), ...
    lx_test_dataset.RandomForest_Prediction(lx_test_dataset.Branch == "PO GORO",:),...
    lx_test_dataset.Lsboost_Prediction(lx_test_dataset.Branch == "PO GORO")],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

tolle_train = array2table([lx_training_dataset.LxObs(lx_training_dataset.Branch == "PO TOLLE",:), ...
    lx_training_dataset.RandomForest_Prediction(lx_training_dataset.Branch == "PO TOLLE",:),...
    lx_training_dataset.Lsboost_Prediction(lx_training_dataset.Branch == "PO TOLLE",:)],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

tolle_test = array2table([lx_test_dataset.LxObs(lx_test_dataset.Branch == "PO TOLLE",:), ...
    lx_test_dataset.RandomForest_Prediction(lx_test_dataset.Branch == "PO TOLLE",:),...
    lx_test_dataset.Lsboost_Prediction(lx_test_dataset.Branch == "PO TOLLE")],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

gnocca_train = array2table([lx_training_dataset.LxObs(lx_training_dataset.Branch == "PO GNOCCA",:), ...
    lx_training_dataset.RandomForest_Prediction(lx_training_dataset.Branch == "PO GNOCCA",:),...
    lx_training_dataset.Lsboost_Prediction(lx_training_dataset.Branch == "PO GNOCCA",:)],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

gnocca_test = array2table([lx_test_dataset.LxObs(lx_test_dataset.Branch == "PO GNOCCA",:), ...
    lx_test_dataset.RandomForest_Prediction(lx_test_dataset.Branch == "PO GNOCCA",:),...
    lx_test_dataset.Lsboost_Prediction(lx_test_dataset.Branch == "PO GNOCCA")],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

dritta_train = array2table([lx_training_dataset.LxObs(lx_training_dataset.Branch == "PO DRITTA",:), ...
    lx_training_dataset.RandomForest_Prediction(lx_training_dataset.Branch == "PO DRITTA",:),...
    lx_training_dataset.Lsboost_Prediction(lx_training_dataset.Branch == "PO DRITTA",:)],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});

dritta_test = array2table([lx_test_dataset.LxObs(lx_test_dataset.Branch == "PO DRITTA",:), ...
    lx_test_dataset.RandomForest_Prediction(lx_test_dataset.Branch == "PO DRITTA",:),...
    lx_test_dataset.Lsboost_Prediction(lx_test_dataset.Branch == "PO DRITTA")],...
    'VariableNames',{'real_lx', 'rf_pred', 'lsb_pred'});


subplot(2,4,1);
plotPerfectFit(goro_train.real_lx, goro_train.rf_pred, goro_test.real_lx,goro_test.rf_pred,"Po Goro", true,30);

subplot(2,4,2);
plotPerfectFit(gnocca_train.real_lx, gnocca_train.rf_pred, gnocca_test.real_lx,gnocca_test.rf_pred,"Po Gnocca", true,30);

subplot(2,4,3);
plotPerfectFit(tolle_train.real_lx, tolle_train.rf_pred, tolle_test.real_lx,tolle_test.rf_pred,"Po Tolle", true,30);

subplot(2,4,4);
plotPerfectFit(dritta_train.real_lx, dritta_train.rf_pred, dritta_test.real_lx,dritta_test.rf_pred,"Po Dritta", true,30);


function [] = plotPerfectFit(obs_train, pred_train, obs_test, pred_test, modelName, addBound, percentageBound)
    if (istable(obs_train))
        obs_train = table2array(obs_train);
    end
    
    if (istable(obs_test))
        obs_test = table2array(obs_test);
    end

    if(istable(pred_train))
        pred_train = table2array(pred_train);
    end

    if(istable(pred_test))
        pred_test = table2array(pred_test);
    end

    hAx=gca;                  
    legendName = {'Train Observations','Test Observations','Perfect prediction'};
    plot(obs_train,pred_train, '.','MarkerSize',18, ...
        'MarkerFaceColor',[0.00,0.45,0.74],'MarkerEdgeColor','auto');
    hold on;
    plot(obs_test,pred_test, '.','MarkerSize',18, ...
        'MarkerFaceColor',[0.93,0.69,0.13],'MarkerEdgeColor','auto');
    xy = linspace(0, 40, 40);
    plot(xy,xy,'k-','LineWidth',1.3);
    if(addBound)
        xyUpperBound = xy + percentageBound*xy/100;
        xyLowerBound = xy - percentageBound*xy/100;
        plot(xy,xyUpperBound, 'r--', 'LineWidth',1.3);
        plot(xy,xyLowerBound, 'r--', 'LineWidth',1.3);
        legendName = {'Train Observations','Test Observations','Perfect prediction', ...
            strcat(string(percentageBound), "% of deviation")};
    end
    hAx.LineWidth=1;
    xlim([0 40]);
    ylim([0 40]);
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
    ylim([0 40]);
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