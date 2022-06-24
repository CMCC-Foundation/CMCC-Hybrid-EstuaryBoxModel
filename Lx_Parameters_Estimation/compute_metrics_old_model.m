%% Add to path subdirectory
addpath(genpath('0-Dataset'));
addpath(genpath('3_Trained-Models'));

%% Create table old model results
algorithm_names = {'old_model' };

results_training = table('Size', [1 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);


results_training("old_model","RMSE") = {computeRMSE(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred)};
results_training("old_model","MAE") = {computeMAE(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred)};
results_training("old_model","RSE") = {computeRSE(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred)};
results_training("old_model","RRSE") = {sqrt(computeRSE(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred))};
results_training("old_model","RAE") = {computeRAE(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred)};
results_training("old_model","R2") = {computeR2(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred)};

corr_coeff_matrix = corrcoef(Old_Model_Training.LX_Obs, Old_Model_Training.LX_Pred);
results_training("old_model","Corr Coeff") = {corr_coeff_matrix(1,2)};

results_test = table('Size', [1 7], ...
    'VariableTypes', {'double','double','double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'RMSE', 'MAE','RSE', 'RRSE','RAE', 'R2', 'Corr Coeff'},...
    'RowNames', algorithm_names);

results_test("old_model","RMSE") = {computeRMSE(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred)};
results_test("old_model","MAE") = {computeMAE(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred)};
results_test("old_model","RSE") = {computeRSE(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred)};
results_test("old_model","RRSE") = {sqrt(computeRSE(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred))};
results_test("old_model","RAE") = {computeRAE(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred)};
results_test("old_model","R2") = {computeR2(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred)};

corr_coeff_matrix = corrcoef(Old_Model_Test.LX_Obs, Old_Model_Test.LX_Pred);
results_test("old_model","Corr Coeff") = {corr_coeff_matrix(1,2)};


writetable(results_training, '3-Trained-Models/Results-calibration-OLD-model.xlsx', 'WriteRowNames',true);
writetable(results_test, '3-Trained-Models/Results-test-OLD-model.xlsx', 'WriteRowNames',true);

function [rmse] = computeRMSE(obs, pred)
    rmse = sqrt(sum((obs - pred).^2)/height(obs));
end

function [mae] = computeMAE(obs, pred)
    mae = (sum(abs(pred-obs)))/height(obs);
end

function [rse] = computeRSE (obs, pred)
    num = sum((pred-obs).^2);
    den = sum((obs-mean(obs)).^2);
    rse = num/den;
end

function [rae] = computeRAE (obs, pred)
    num = sum(abs(pred-obs));
    den = sum(abs(mean(obs) - obs));
    rae = num / den;
end

function [r2] = computeR2 (obs, pred)
    sse = sum((obs-pred).^2);
    sst = sum((obs - mean(obs)).^2);
    r2 = 1 - (sse/sst);
end


