function plot_bar_q_river_splitted_obs_by_year(dataset)
    year_obs = unique(dataset.Year);
    result = zeros(4,2);
    
    for i = 1:numel(year_obs)
        % Q_River_LOW
        result(i,1) = height(dataset(dataset.Year == year_obs(i) & strcmp(string(dataset.QriverClass), "LOW"),:));

        % Q_River_STRONG
        result(i,2) = height(dataset(dataset.Year == year_obs(i) & strcmp(string(dataset.QriverClass), "STRONG"),:));
    end

    figure();
    hAx=gca;
    bar(result);
    hAx.LineWidth=1;
    ylim([0 500]);
    xlabel('Year');
    ylabel('Observations');
    xticks(linspace(1, numel(year_obs), numel(year_obs)));
    xticklabels(string(year_obs));
    title('Number of observations by year for q-river-class');
    legend('Low','Strong','Location','northeast');
    set(gca,'FontSize',12);
end