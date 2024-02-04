function [f] = create_perfect_fit(resumePredictions,algorithm_names,addBoundPerfectFit, percentageBoundPerfectFit)
%CREATE_PERFECT_FIT This function plot a perfect predictions plot
%   Input:
%   1) resumePredictions - table with a summary of observed and predicted
%   values
%   2) algorithm_names - string array with the names of the trained models
%   3) addBoundPerfectFit - boolean value to add or not a bound on the
%   perfect predictions line
%   4) percentageBoundPerfectFit - percentage bound to be added on the
%   perfect predictions plot

    f = figure;
    %tiledlayout(2,2);
    %f.Position = [0 -5 1068 1001];
    f.Position = [73,72,1631,852];
    tiledlayout(3,3);
    %f.Position = [129,46,1532,460];
    for i = 1:numel(algorithm_names)
        nexttile
        plotPerfectFit( ...
            resumePredictions(:,1), ...
            resumePredictions(:,i+1), ...
            algorithm_names(i), ...
            addBoundPerfectFit, ...
            percentageBoundPerfectFit);
    end
end