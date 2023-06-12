function plot_boxplot_training_test(subplot_title, varargin)
%PLOT_BOXPLOT_TRAINING_TEST This function plot a barchart for the feature importance
%score
%   Input:
%   1) subplot_title - array of features name
%  
%   2) varargin - an array with variable argument

    n_features = width(varargin{1});
    n_col_subplot = n_features./2;

    n_rows_dat_1 = height(varargin{1});
    n_rows_dat_2 = height(varargin{2});

    figure();

    for i = 1:n_features
        features_dat_1 = table2array(varargin{1}(:,i));
        features_dat_2 = table2array(varargin{2}(:,i));
        features = [features_dat_1; features_dat_2];

        g1 = repmat({'Training dataset'},(n_rows_dat_1),1);
        g2 = repmat({'Test dataset'},n_rows_dat_2,1);       
        g = [g1; g2];
        
        subplot(round(n_col_subplot),2,i);
        hAx=gca;                  
        boxplot(features, g);
        hAx.LineWidth=1;
        title(varargin{1}(:,i).Properties.VariableNames);
    end
    sgtitle(subplot_title);
end