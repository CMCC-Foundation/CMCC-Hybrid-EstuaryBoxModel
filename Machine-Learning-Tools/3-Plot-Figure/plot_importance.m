function plot_importance (feature_importance,nameDataset)
	sortedFeatureImportance = sortrows(feature_importance,'score','descend');
	sortedFeatureImportance = sortedFeatureImportance(1:end,:);
	
    titles = categorical(sortedFeatureImportance.Properties.RowNames);
    titles = reordercats(titles,sortedFeatureImportance.Properties.RowNames);

	bar(titles,sortedFeatureImportance.score);

	title(nameDataset);
	set(gca,'TickLabelInterpreter','none');
end