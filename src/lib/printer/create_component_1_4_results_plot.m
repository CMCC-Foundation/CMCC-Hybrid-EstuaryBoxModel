function [] = create_component_1_4_results_plot(resumePredictions,algorithm_names, ...
    addBoundPerfectFit, percentageBoundPerfectFit, xylim, response, tr_te_dataset, ncomp)
%CREATE_COMPONENT_1_4_RESULTS_PLOT This function plot a perfect predictions plot
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
%   8) ncomp - the number of component of Hyb-EBM: 1 for Lx, 4 for Sul
    
    if (ncomp==1)
        path_fig = "..\..\..\reports\figures\Component-1-Lx\";
        name_comp_fig = "Component-1-Lx";
    else
        path_fig = "..\..\..\reports\figures\Component-4-Sul\";
        name_comp_fig = "Component-4-Sul";
    end
    
    f = figure;
    f.Position = [0 0 1500 450];
    n_alg = numel(algorithm_names);

    for i = 1:n_alg
        subplot(1,3,i);
        plot_perfect_fit( resumePredictions(:,1), resumePredictions(:,i+1), ...
            algorithm_names(i), addBoundPerfectFit, percentageBoundPerfectFit, xylim);
    end
    saveas(gcf,strcat(path_fig, name_comp_fig,"-Perfect-Fit-Plot-", tr_te_dataset,".png"));
    
    f = figure;
    f.Position = [0 0 1500 450];

    for i = 1:n_alg
        subplot(1,3,i);
        resumeTable = create_residuals_table(resumePredictions(:,1), ...
            resumePredictions(:,i+1), ...
            response);
        plot_residuals_bar(resumeTable, algorithm_names(i), response, xylim);
    end
    saveas(gcf,strcat(path_fig, name_comp_fig,"-Response-Plot-", tr_te_dataset,".png"));
end