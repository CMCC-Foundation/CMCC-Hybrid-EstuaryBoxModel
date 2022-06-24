addpath(genpath("..\3_Trained-Models\Iteration-Results"));
load("..\3-Trained-Models\Iteration-Results\results-iteration.mat");

%% Training
subplot(1,2,1);
training_rf_mean = mean(table2array(results_training_rf),1);
training_rf_dv = std(table2array(results_training_rf),0,1);

training_lsb_mean = mean(table2array(results_training_lsb),1);
training_lsb_dv = std(table2array(results_training_lsb),0,1);

plotErrorBar(training_rf_mean, training_rf_dv, training_lsb_mean, training_lsb_dv, 'Training on 5 iteration');

%% Test
subplot(1,2,2);

test_rf_mean = mean(table2array(results_test_rf),1);
test_rf_dv = std(table2array(results_test_rf),0,1);

test_lsb_mean = mean(table2array(results_test_lsb),1);
test_lsb_dv = std(table2array(results_test_lsb),0,1);

plotErrorBar(test_rf_mean, test_rf_dv, test_lsb_mean, test_lsb_dv, 'Test');

function [] = plotErrorBar (data1, err1, data2, err2, titleName)
    errorbar(data1,err1, 'vertical','.');
    hold on;
    errorbar(data2, err2,  'vertical','.');
    hold off;
    xlim([0.9 7.1]);
    xticks([1 2 3 4 5 6 7]);
    xticklabels({'RMSE','MAE', 'RSE', 'RRSE', 'RAE', 'R2', 'CorrCoeff'});
    legend({'Random forest','Lsboost'});
    title(titleName);
end
