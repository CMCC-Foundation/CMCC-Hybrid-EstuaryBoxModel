function plot_scatterplot(features, target)
featuresName = features.Properties.VariableNames;
targetName = target.Properties.VariableNames;
figure();
for i=1:numel(featuresName)
    subplot(1,numel(featuresName),i);
    hAx=gca;
    scatter(table2array(features(:,i)),table2array(target(:,"Lx_OBS")));
    xlabel(featuresName(i))
    ylabel(targetName)
    hAx.LineWidth=1;
    set(gca,'FontSize',12);
    grid on
end
    sgtitle("Scatter plot between input and target features for Lx dataset");
end