function [resumeTable] = create_residuals_table(obs, pred, response)
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