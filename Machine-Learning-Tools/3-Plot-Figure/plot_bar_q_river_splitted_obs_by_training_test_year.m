function plot_bar_q_river_splitted_obs_by_training_test_year(dataset)
    result = zeros(4,2);
    
    % training Q_River_LOW
    result(1,1) = height(dataset((dataset.Year == 2016 | dataset.Year == 2017) ...
        & strcmp(string(dataset.QriverClass), "LOW"),:));
    % training Q_River_STRONG
    result(1,2) = height(dataset((dataset.Year == 2016 | dataset.Year == 2017) ...
        & strcmp(string(dataset.QriverClass), "STRONG"),:));

    % test 2018 Q_River_LOW
    result(2,1) = height(dataset((dataset.Year == 2018) ...
        & strcmp(string(dataset.QriverClass), "LOW"),:));
    % test 2018 Q_River_STRONG
    result(2,2) = height(dataset((dataset.Year == 2018) ...
        & strcmp(string(dataset.QriverClass), "STRONG"),:));

        % test 2019 Q_River_LOW
    result(3,1) = height(dataset((dataset.Year == 2019) ...
        & strcmp(string(dataset.QriverClass), "LOW"),:));
    % test 2019 Q_River_STRONG
    result(3,2) = height(dataset((dataset.Year == 2019) ...
        & strcmp(string(dataset.QriverClass), "STRONG"),:));
    
    figure();
    hAx=gca;
    bar(result);
    hAx.LineWidth=1;
    ylim([0 500]);
    xlabel('Year');
    ylabel('Observations');
    xticks([1,2,3]);
    xticklabels({'Training 2016-2017', 'Test 2018', 'Test 2019'});
    title('Number of observations by training/test for q-river-class');
    legend('Low','Strong','Location','northeast');
    set(gca,'FontSize',12);
end