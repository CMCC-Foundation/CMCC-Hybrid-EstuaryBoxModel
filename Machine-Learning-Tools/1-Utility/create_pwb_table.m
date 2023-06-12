function [pwbTable] = create_pwb_table(obs,pred, pwbTable,algorithm_name, pwbX)
%CREATE_PWB_TABLE This function compute the pwb Table
%   Input:
%
%   1) obs - the observed values
%
%   2) pred - the predicted values from regression model
%
%   3) pwbTable - empty table in which save the results
%   
%   4) algorithm_name - the name of the algorithm for which we want to
%   compute the pwbTabel
%
%   5) pwbX - the different threshold in the pwbTable
%
%   Output:
%
%   1) pwbTable - the table with the results updated

    if istable(obs)
        obs = table2array(obs);
    end
    if istable(pred)
        pred = table2array(pred);
    end
    for i=1:height(pwbTable)
        pwbTable(i,algorithm_name) = {round(computePWBTable(obs, pred, pwbX(i)),2)};
    end
end
    
function [pwbTest] = computePWBTable (obs, pred, threshold)
    minBound = obs - (obs*threshold/100);
    maxBound = obs + (obs*threshold/100);
    countInBound = sum(pred>= minBound & pred<=maxBound);
    pwbTest = countInBound*100/numel(obs);
end
