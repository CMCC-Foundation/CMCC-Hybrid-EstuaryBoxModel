function [resumeTable] = create_residuals_table(obs, pred, response)
%CREATE_RESIDUALS_TABLE This function create the resumeTable required by
%PLOT_RESIDUALS_BAR.
%   Input:
%   1) obs - vector with the observed data
%   2) pred - vctor with the predicted data
%   3) response - the name of the target variable

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