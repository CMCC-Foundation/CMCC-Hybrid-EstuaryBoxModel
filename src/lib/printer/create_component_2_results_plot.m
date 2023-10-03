function [] = create_component_2_results_plot(resumePredictions,algorithm_names, ...
    addBoundPerfectFit, percentageBoundPerfectFit, xylim, response, tr_te_dataset)
%CREATE_COMPONENT_2_RESULTS_PLOT This function plot a perfect predictions plot
%   Input:
%   1) resumePredictions - table with a summary of observed and predicted
%   values
%   2) algorithm_names - string array with the names of the trained models
%   3) addBoundPerfectFit - boolean value to add or not a bound on the
%   perfect predictions line
%   4) percentageBoundPerfectFit - percentage bound to be added on the
%   perfect predictions plot
%   5) xylim - the max min on x/y axes
%   6) response - the name of the target value
%   7) tr_te_dataset - used to specify if we are plotting training or test
%   results
    
    f = figure;
    f.Position = [0 0 1140 450];
    nMax = width(resumePredictions);
    j = 1;
    for i = 1:2:nMax
        subplot(1,2,j);
        plot_perfect_fit(resumePredictions(:,i), resumePredictions(:,i+1), ...
            algorithm_names(j), addBoundPerfectFit, percentageBoundPerfectFit, xylim);
        j=j+1;
    end
    saveas(gcf,strcat("..\..\..\reports\figures\Component-2-Ck\", ...
        "Component-2-Ck-Perfect-Fit-Plot-", tr_te_dataset,".png"));

    f = figure;
    f.Position = [0 0 1140 450];
    j = 1;
    for i = 1:2:nMax
        subplot(1,2,j);
        resumeTable = create_residuals_table(resumePredictions(:,i),resumePredictions(:,i+1), response);
        plot_residuals_bar( resumeTable, algorithm_names(j), response, xylim);
        j=j+1;
    end
    saveas(gcf,strcat("..\..\..\reports\figures\Component-2-Ck\", ...
        "Component-2-Ck-Response-Plot-", tr_te_dataset,".png"));
end