function plot_corrplot(dataset)
    figure();
    hAx=gca;
    corrplot(dataset,DataVariables=dataset.Properties.VariableNames);
    hAx.LineWidth=1;
    title('Correlation matrix of salinity dataset');
    set(gca,'FontSize',12);
end