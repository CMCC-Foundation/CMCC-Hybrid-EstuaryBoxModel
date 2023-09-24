function [] = create_component_1_results_plot(resumePredictions,algorithm_names, ...
    addBoundPerfectFit, percentageBoundPerfectFit, xylim, response)
%CREATE_PERFECT_FIT This function plot a perfect predictions plot
%   Input:
%   1) resumePredictions - table with a summary of observed and predicted
%   values
%  
%   2) algorithm_names - string array with the names of the trained models
%   
%   3) addBoundPerfectFit - boolean value to add or not a bound on the
%   perfect predictions line
%
%   4) percentageBoundPerfectFit - percentage bound to be added on the
%   perfect predictions plot
    f = figure;
    f.Position = [0 0 1500 450];
    
    n_alg = numel(algorithm_names);

    for i = 1:n_alg
        subplot(1,3,i);
        plot_perfect_fit( resumePredictions(:,1), resumePredictions(:,i+1), ...
            algorithm_names(i), addBoundPerfectFit, percentageBoundPerfectFit, xylim);
    end
    
    f = figure;
    f.Position = [0 0 1500 450];

    for i = 1:n_alg
        subplot(1,3,i);
        resumeTable = create_residuals_table(resumePredictions(:,1), ...
            resumePredictions(:,i+1), ...
            response);
        plot_residuals_bar(resumeTable, algorithm_names(i), response, xylim);
    end

end