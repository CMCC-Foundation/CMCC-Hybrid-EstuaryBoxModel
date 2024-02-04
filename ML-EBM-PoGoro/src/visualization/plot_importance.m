function f = plot_importance (feature_importance, nameDataset)
%PLOT_IMPORTANCE This function plot a barchart for the feature importance
%score
%   Input:
%   1) feature_importance - the feature importance table
%  
%   2) nameDataset - the name of the dataset for which the features
%   importance has been computed
    f = figure();
	sortedFeatureImportance = sortrows(feature_importance,'score','descend');
	sortedFeatureImportance = sortedFeatureImportance(1:end,:);
	
    titles = categorical(sortedFeatureImportance.Properties.RowNames);
    titles = reordercats(titles,sortedFeatureImportance.Properties.RowNames);

	bar(titles,sortedFeatureImportance.score);

	title(nameDataset);
	set(gca,'TickLabelInterpreter','Tex');
    xlabel("Feature");
    ylabel("Feature Importance")
end