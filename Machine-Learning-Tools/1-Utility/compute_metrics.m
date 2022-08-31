function [results] = compute_metrics(obs, pred, algorithm_names, results)
%COMPUTE_METRICS This function compute 8 different metrics to evaluate regression models performance
%   obs: real values
%   pred: predicted values from regression model
%   algorithm_names: names of the regression model used
%   results: table to store performance
        
    if(istable(obs))
        obs = table2array(obs);
    end

    if(istable(pred))
        pred = table2array(pred);
    end
    
    results(algorithm_names,'RMSE') = {computeRMSE(obs, pred)}; 
    results(algorithm_names,'NRMSE') = {computeNRMSE(obs, pred)}; 
    results(algorithm_names,'MAE') = {computeMAE(obs, pred)}; 
    results(algorithm_names,'RSE') = {computeRSE(obs, pred)}; 
    results(algorithm_names,'RRSE') = {sqrt(computeRSE(obs, pred))}; 
    results(algorithm_names,'RAE') = {computeRAE(obs, pred)}; 
    results(algorithm_names,'R2') = {computeR2(obs, pred)}; 
    results(algorithm_names,'Corr Coeff') = {computeCorrCoef(obs, pred)}; 
end

function [rmse] = computeRMSE(obs, pred)
    rmse = sqrt(sum((obs - pred).^2)/height(obs));
end

function [nrmse] = computeNRMSE(obs, pred)
    nrmse = computeRMSE(obs, pred) / mean(obs);
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

function [r] = computeCorrCoef(obs, pred)
    corr_coeff_matrix = corrcoef(obs, pred);
    r = corr_coeff_matrix(1,2);
end