function [] = plotTimeseries(obs, pred, title_timeseries)
    if (istable(obs))
        obs = table2array(obs);
    end
    
    if(istable(pred))
        pred = table2array(pred);
    end

    hAx=gca; 
    plot(obs, 'LineWidth', 1.2, 'Color', [0.00,0.45,0.74]);
    hold on
    plot(pred, 'LineWidth', 1.2, 'Color', [1.00,0.00,0.00]);
    legend(["Observations", "LSTM predictions"]);
    title(title_timeseries);
    xlabel("Time steps");
    ylabel("Sul (psu)");
    hAx.LineWidth=1.4;
    ylim([0 30]);
    yticks([0 5 10 15 20 25 30 35 40]);
    yticklabels({'0','5','10','15','20','25','30','35','40'});
    set(gca,'FontSize',14);
    grid on;
end