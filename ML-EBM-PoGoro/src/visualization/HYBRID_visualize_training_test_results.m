
addpath(genpath("..\lib\"));

%% Read dataset
algorithm_names = {'EBM', 'Hybrid-EBM-RF','Hybrid-EBM-LSBoost', 'RF-no-Season', 'LSBoost-no-Season', 'NN-no-Season',...
    'RF-Season', 'LSBoost-Season', 'NN-Season'};
%algorithm_names = {'RF', 'LSBoost', 'NN'};
response = 'Sul';

%% Training dataset
%training_table_results = models_training_predictions(:, ["Sul","Sul_RF", "Sul_LSBoost","Sul_NN"]);
%f = create_perfect_fit(training_table_results,algorithm_names,true,30);
%saveas(f, "..\..\reports\figure\Season-Feature\Perfect-Fit-Plot-Train.fig");
%exportgraphics(f, "..\..\reports\figure\Season-Feature\Perfect-Fit-Plot-Train.jpg","BackgroundColor","white", "Resolution", 600);

%{
f = create_residuals_plot(training_table_results,algorithm_names,response);
saveas(f, "..\..\reports\figure\Season-Feature\Response-Plot-Train.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Response-Plot-Train.jpg","BackgroundColor","white", "Resolution", 600);
%}

%% Test dataset
%test_table_results = models_test_predictions(:, ["Sul","Sul_EBM", "Sul_Hybrid_EBM_RF","Sul_Hybrid_EBM_LSBoost","Sul_RF", "Sul_LSBoost","Sul_NN"]);
test_table_results = models_test_predictions;
f = create_perfect_fit(test_table_results,algorithm_names,true,30);
saveas(f, "D:\4_CMCC\Progetti\3_EBM_EstuarIO\1-Data-Obs-Arpae\2_EBM-Only-ML-Po-Goro\Predictions-Compare-Hybrid\Perfect-Fit-Plot-Test-NO-SEASON.fig");
exportgraphics(f, "D:\4_CMCC\Progetti\3_EBM_EstuarIO\1-Data-Obs-Arpae\2_EBM-Only-ML-Po-Goro\Predictions-Compare-Hybrid\Perfect-Fit-Plot-Test-NO-SEASON.jpg", ...
    "BackgroundColor","white", "Resolution", 600);

%{
f = create_residuals_plot(test_table_results,algorithm_names,response);
saveas(f, "..\..\reports\figure\Season-Feature\Response-Plot-Test.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Response-Plot-Test.jpg","BackgroundColor","white", "Resolution", 600);
%}


