function [pwbTable] = create_pwb_table(obs,pred, pwbTable,algorithm_name, pwbX)
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
