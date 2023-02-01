function plot_boxplot(subplot_title, varargin)
    n_features = width(varargin{1});
    n_col_subplot = round(n_features./2);

    n_rows_dat_1 = height(varargin{1});
    n_rows_dat_2 = height(varargin{2});
    n_rows_dat_3 = height(varargin{3});
    n_rows_dat_4 = height(varargin{4});

    figure();

    for i = 1:n_features
        features_dat_1 = table2array(varargin{1}(:,i));
        features_dat_2 = table2array(varargin{2}(:,i));
        features_dat_3 = table2array(varargin{3}(:,i));
        features_dat_4 = table2array(varargin{4}(:,i));
        features = [features_dat_1; features_dat_2; features_dat_3; features_dat_4];

        g1 = repmat({'Training 2016-2017'},(n_rows_dat_1),1);
        g2 = repmat({'Test-2018'},n_rows_dat_2,1);
        g3 = repmat({'Test-2019'},n_rows_dat_3,1);
        g4 = repmat({'Test-2018-2019'},(n_rows_dat_4),1);        
        g = [g1; g2; g3; g4];
        
        subplot(n_col_subplot,2,i);
        hAx=gca;                  
        boxplot(features, g);
        hAx.LineWidth=1;
        title(varargin{1}(:,i).Properties.VariableNames);
    end
    sgtitle(subplot_title);
end

