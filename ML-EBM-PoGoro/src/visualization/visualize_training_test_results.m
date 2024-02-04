clc
%clear
%close all

addpath(genpath("..\lib\"));

%% Read dataset
%{
models_training_predictions = import_dataset("..\..\models\Season-Feature\models-predictions.xlsx", 7, ...
    "A2:G963", "training-predictions", ...
    ["ID","Date","Sul","Sul_EBM", "Sul_RF", "Sul_LSBoost", "Sul_NN" ], ...
    ["categorical","datetime", "double", "double", "double", "double", "double"]);

models_test_predictions = import_dataset("..\..\models\Season-Feature\models-predictions.xlsx", 7, ...
    "A2:G238", "test-predictions", ...
    ["ID","Date","Sul", "Sul_EBM", "Sul_RF", "Sul_LSBoost", "Sul_NN" ], ...
    ["categorical","datetime", "double", "double", "double", "double", "double"]);
%}
%algorithm_names = {'EBM', 'RF', 'LSBoost', 'NN'};
algorithm_names = {'RF', 'LSBoost', 'NN','RF', 'LSBoost', 'NN'};

response = 'Sul';

%% Training dataset
%training_table_results = models_training_predictions(:, ["Sul", "Sul_RF","Sul_LSBoost","Sul_NN", "Sul_RF_seas","Sul_LSBoost_seas","Sul_NN_seas"]);
%f = create_perfect_fit(training_table_results,algorithm_names,true,30);
%saveas(f, "..\..\reports\figure\Season-Feature\Perfect-Fit-Plot-Train.fig");
%exportgraphics(f, "..\..\reports\figure\Season-Feature\Perfect-Fit-Plot-Train.jpg","BackgroundColor","white", "Resolution", 600);

%{
f = create_residuals_plot(training_table_results,algorithm_names,response);
saveas(f, "..\..\reports\figure\Season-Feature\Response-Plot-Train.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Response-Plot-Train.jpg","BackgroundColor","white", "Resolution", 600);
%}

%% Test dataset
test_table_results = models_test_predictions(:, ["Sul","Sul_RF", "Sul_LSBoost","Sul_NN"]);
f = create_perfect_fit(test_table_results,algorithm_names,true,30);
saveas(f, "..\..\reports\figure\Season-Feature\Perfect-Fit-Plot-Test.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Perfect-Fit-Plot-Test.jpg","BackgroundColor","white", "Resolution", 600);

%{
f = create_residuals_plot(test_table_results,algorithm_names,response);
saveas(f, "..\..\reports\figure\Season-Feature\Response-Plot-Test.fig");
exportgraphics(f, "..\..\reports\figure\Season-Feature\Response-Plot-Test.jpg","BackgroundColor","white", "Resolution", 600);
%}