addpath(genpath("..\1_Trained-Models\Iteration-Results"));
load("..\1-Trained-Models\Iteration-Results\results-iteration.mat");

metrics =  {'RMSE','NRMSE','MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'};
rowNames = {'Mean','Standard Deviation'};

%% Training
subplot(1,2,1);
training_rf_mean = mean(table2array(results_iteration.results_training_rf),1);
training_rf_dv = std(table2array(results_iteration.results_training_rf),0,1);

training_lsb_mean = mean(table2array(results_iteration.results_training_lsb),1);
training_lsb_dv = std(table2array(results_iteration.results_training_lsb),0,1);

plotErrorBar(training_rf_mean, training_rf_dv, training_lsb_mean, training_lsb_dv, 'Training on 5 iteration');

rf_training_table = array2table([training_rf_mean;training_rf_dv] ,"VariableNames", metrics,"RowNames",rowNames);
lsb_training_table = array2table([training_lsb_mean;training_lsb_dv] ,"VariableNames", metrics,"RowNames",rowNames);

%% Test
subplot(1,2,2);

test_rf_mean = mean(table2array(results_iteration.results_test_rf),1);
test_rf_dv = std(table2array(results_iteration.results_test_rf),0,1);

test_lsb_mean = mean(table2array(results_iteration.results_test_lsb),1);
test_lsb_dv = std(table2array(results_iteration.results_test_lsb),0,1);

plotErrorBar(test_rf_mean, test_rf_dv, test_lsb_mean, test_lsb_dv, 'Test');

rf_test_table = array2table([test_rf_mean;test_rf_dv] ,"VariableNames", metrics,"RowNames",rowNames);
lsb_test_table = array2table([test_lsb_mean;test_lsb_dv] ,"VariableNames", metrics,"RowNames",rowNames);

results_mean_dev = struct("rf_training_table",rf_training_table,"lsb_training_table",lsb_training_table,...
    "rf_test_table",rf_test_table,"lsb_test_table",lsb_test_table);
save("..\1-Trained-Models\Iteration-Results\results-mean-dev.mat", "results_mean_dev");

function [] = plotErrorBar (data1, err1, data2, err2, titleName)
    errorbar(data1,err1, 'vertical','.','LineWidth',1,'CapSize',10,MarkerSize=10);
    hold on;
    errorbar(data2, err2,  'vertical','.','LineWidth',1,'CapSize',10,MarkerSize=10);
    hold off;
    xlim([0 9]);
    xticks([0 1 2 3 4 5 6 7 8 9 ]);
    xticklabels({'','RMSE','NRMSE', 'MAE', 'RSE', 'RRSE', 'RAE', 'R2', 'CorrCoeff',''});
    legend({'Random forest','Lsboost'});
    title(titleName);
end
