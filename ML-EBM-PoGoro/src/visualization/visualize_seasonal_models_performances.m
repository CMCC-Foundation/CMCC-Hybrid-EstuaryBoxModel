clc
clear
close all
addpath(genpath("..\..\data\"));
addpath(genpath("..\lib"));

%% Read dataset in which 'Season' is not a feature
models_test_predictions = import_dataset("..\..\models\models-predictions.xlsx", 7, ...
    "A2:G238", "test-predictions", ...
    ["ID","Date","Sul", "Sul_EBM", "Sul_RF", "Sul_LSBoost", "Sul_NN" ], ...
    ["categorical","datetime", "double", "double", "double", "double", "double"]);

models_test_predictions = split_date_in_season(models_test_predictions, unique(year(models_test_predictions.Date)));
models_test_predictions = models_test_predictions(:,["Sul", "Sul_EBM", "Sul_RF", "Sul_LSBoost", "Sul_NN","Season"]);
%{
%% ===================== SEASON IS NOT FEATURE ============================

%% Plot comparison errorbar Mean,Stdv SulObs / Sul_EBM / Sul_RF
stats_pred_models = grpstats(models_test_predictions,"Season", ["mean","std"]);
figure();
x = [1, 2, 3, 4];
errorbar(x, stats_pred_models.mean_Sul, stats_pred_models.std_Sul,"-s","MarkerSize",10,...
    "Color", "blue", "MarkerEdgeColor","blue","MarkerFaceColor",[0.65 0.85 0.90], "LineStyle","none", "LineWidth", 1.2);
hold on;
errorbar(x-0.2, stats_pred_models.mean_Sul_EBM, stats_pred_models.std_Sul_EBM,"-s","MarkerSize",10,...
    "Color", [0.93,0.69,0.13], "MarkerEdgeColor",[0.93,0.69,0.13],"MarkerFaceColor",[1.00,1.00,0.07], "LineStyle","none", "LineWidth", 1.2);
errorbar(x+0.2, stats_pred_models.mean_Sul_RF, stats_pred_models.std_Sul_RF,"-s","MarkerSize",10,...
    "Color", [1.00,0.00,0.00], "MarkerEdgeColor",[1.00,0.00,0.00],"MarkerFaceColor", [0.85,0.33,0.10], "LineStyle","none", "LineWidth", 1.2);
legend(["Observed", "EBM", "RF"]);
xticks(x);
xticklabels(stats_pred_models.Season);
yticks([0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25]);
xlim([0.5, 4.5]);
ylim([0,26]);
xlabel("Season");
ylabel("Sul (psu)");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-season-errorbar-RF-EBM-test.png");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-season-errorbar-RF-EBM-test.fig");

%% Plot comparison errorbar rmse,Stdv SulObs / Sul_EBM / Sul_RF / Sul_LSBoost without season feature
season = string(unique(models_test_predictions.Season));
grp_stat = table('Size', [4 6], ...
    'VariableTypes', repmat({'double'}, 1, 6), ...
    'VariableNames', {'rmse_Sul_EBM', 'std_Sul_EBM', 'rmse_Sul_RF', 'std_Sul_RF', 'rmse_Sul_LSBoost', 'std_Sul_LSBoost'},...
    'RowNames', {'Autumn', 'Spring', 'Winter', 'Summer'});

for i=1:numel(season)
    tb = models_test_predictions(models_test_predictions.Season == season(i),:);
    [rmse, stdev] = computeRMSE_Stdev(tb(:,1), tb(:,2));
    grp_stat(season(i), "rmse_Sul_EBM") = {rmse};
    grp_stat(season(i), "std_Sul_EBM") = {stdev};

    [rmse, stdev] = computeRMSE_Stdev(tb(:,1), tb(:,3));
    grp_stat(season(i), "rmse_Sul_RF") = {rmse};
    grp_stat(season(i), "std_Sul_RF") = {stdev};

    [rmse, stdev] = computeRMSE_Stdev(tb(:,1), tb(:,4));
    grp_stat(season(i), "rmse_Sul_LSBoost") = {rmse};
    grp_stat(season(i), "std_Sul_LSBoost") = {stdev};
end

figure();
x = [1, 2, 3, 4];
hold on;
errorbar(x-0.2, grp_stat.rmse_Sul_EBM, grp_stat.std_Sul_EBM,"-s","MarkerSize",10,...
    "Color", [0.93,0.69,0.13], "MarkerEdgeColor",[0.93,0.69,0.13],"MarkerFaceColor",[1.00,1.00,0.07], "LineStyle","none", "LineWidth", 1.2);
errorbar(x, grp_stat.rmse_Sul_RF, grp_stat.std_Sul_RF,"-s","MarkerSize",10,...
    "Color", "blue", "MarkerEdgeColor","blue","MarkerFaceColor",[0.65 0.85 0.90], "LineStyle","none", "LineWidth", 1.2);
errorbar(x+0.2, grp_stat.rmse_Sul_LSBoost, grp_stat.std_Sul_LSBoost,"-s","MarkerSize",10,...
    "Color", [1.00,0.00,0.00], "MarkerEdgeColor",[1.00,0.00,0.00],"MarkerFaceColor", [0.85,0.33,0.10], "LineStyle","none", "LineWidth", 1.2);
legend(["EBM", "RF", "LSBoost"]);
xticks(x);
xticklabels(season);
yticks([0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25]);
xlim([0.5, 4.5]);
ylim([-5,15]);
xlabel("Season");
ylabel("Sul (psu)");
%}

%% ===================== SEASON IS FEATURE ============================
%% Read dataset in which 'Season' is a feature
models_test_predictions = import_dataset("..\..\models\Season-Feature\models-predictions.xlsx", 7, ...
    "A2:G238", "test-predictions", ...
    ["ID","Date","Sul", "Sul_EBM", "Sul_RF", "Sul_LSBoost", "Sul_NN" ], ...
    ["categorical","datetime", "double", "double", "double", "double", "double"]);

models_test_predictions = split_date_in_season(models_test_predictions, unique(year(models_test_predictions.Date)));
models_test_predictions = models_test_predictions(:,["Sul", "Sul_EBM", "Sul_RF", "Sul_LSBoost", "Sul_NN","Season"]);

%% Plot comparison errorbar Mean,Stdv SulObs / Sul_EBM / Sul_RF / Sul_LSBoost with season feature
%{
stats_pred_models = grpstats(models_test_predictions,"Season", ["mean","std"]);
figure();
x = [1, 2, 3, 4];
errorbar(x, stats_pred_models.mean_Sul, stats_pred_models.std_Sul,"-s","MarkerSize",10,...
    "Color", "blue", "MarkerEdgeColor","blue","MarkerFaceColor",[0.65 0.85 0.90], "LineStyle","none", "LineWidth", 1.2);
hold on;
errorbar(x-0.2, stats_pred_models.mean_Sul_EBM, stats_pred_models.std_Sul_EBM,"-s","MarkerSize",10,...
    "Color", [0.93,0.69,0.13], "MarkerEdgeColor",[0.93,0.69,0.13],"MarkerFaceColor",[1.00,1.00,0.07], "LineStyle","none", "LineWidth", 1.2);
errorbar(x+0.2, stats_pred_models.mean_Sul_RF, stats_pred_models.std_Sul_RF,"-s","MarkerSize",10,...
    "Color", [1.00,0.00,0.00], "MarkerEdgeColor",[1.00,0.00,0.00],"MarkerFaceColor", [0.85,0.33,0.10], "LineStyle","none", "LineWidth", 1.2);
errorbar(x+0.4, stats_pred_models.mean_Sul_LSBoost, stats_pred_models.std_Sul_LSBoost,"-s","MarkerSize",10,...
    "Color", [1.00,0.00,0.00], "MarkerEdgeColor",[1.00,0.00,0.00],"MarkerFaceColor", [0.85,0.33,0.10], "LineStyle","none", "LineWidth", 1.2);
legend(["Observed", "EBM", "RF", "LSBoost"]);
xticks(x);
xticklabels(stats_pred_models.Season);
yticks([0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25]);
xlim([0.5, 4.5]);
ylim([0,26]);
xlabel("Season");
ylabel("Sul (psu)");
%}

%% Plot comparison errorbar rmse,Stdv SulObs / Sul_EBM / Sul_RF / Sul_LSBoost with season feature
season = string(unique(models_test_predictions.Season));
grp_stat = table('Size', [4 6], ...
    'VariableTypes', repmat({'double'}, 1, 6), ...
    'VariableNames', {'rmse_Sul_EBM', 'std_Sul_EBM', 'rmse_Sul_RF', 'std_Sul_RF', 'rmse_Sul_LSBoost', 'std_Sul_LSBoost'},...
    'RowNames', {'Autumn', 'Spring', 'Winter', 'Summer'});

for i=1:numel(season)
    tb = models_test_predictions(models_test_predictions.Season == season(i),:);
    [rmse, stdev] = computeRMSE_Stdev(tb(:,1), tb(:,2));
    grp_stat(season(i), "rmse_Sul_EBM") = {rmse};
    grp_stat(season(i), "std_Sul_EBM") = {stdev};

    [rmse, stdev] = computeRMSE_Stdev(tb(:,1), tb(:,3));
    grp_stat(season(i), "rmse_Sul_RF") = {rmse};
    grp_stat(season(i), "std_Sul_RF") = {stdev};

    [rmse, stdev] = computeRMSE_Stdev(tb(:,1), tb(:,4));
    grp_stat(season(i), "rmse_Sul_LSBoost") = {rmse};
    grp_stat(season(i), "std_Sul_LSBoost") = {stdev};
end

figure();
x = [1, 2, 3, 4];
hold on;
errorbar(x-0.2, grp_stat.rmse_Sul_EBM, grp_stat.std_Sul_EBM,"-s","MarkerSize",10,...
    "Color", [0.93,0.69,0.13], "MarkerEdgeColor",[0.93,0.69,0.13],"MarkerFaceColor",[1.00,1.00,0.07], "LineStyle","none", "LineWidth", 1.2);
errorbar(x, grp_stat.rmse_Sul_RF, grp_stat.std_Sul_RF,"-s","MarkerSize",10,...
    "Color", "blue", "MarkerEdgeColor","blue","MarkerFaceColor",[0.65 0.85 0.90], "LineStyle","none", "LineWidth", 1.2);
errorbar(x+0.2, grp_stat.rmse_Sul_LSBoost, grp_stat.std_Sul_LSBoost,"-s","MarkerSize",10,...
    "Color", [1.00,0.00,0.00], "MarkerEdgeColor",[1.00,0.00,0.00],"MarkerFaceColor", [0.85,0.33,0.10], "LineStyle","none", "LineWidth", 1.2);
legend(["EBM", "RF", "LSBoost"]);
xticks(x);
xticklabels(season);
yticks([0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25]);
xlim([0.5, 4.5]);
ylim([-5,15]);
xlabel("Season");
ylabel("Sul (psu)");

function [rmse, stdv] = computeRMSE_Stdev(obs, pred)
    if istable(obs)
        obs = table2array(obs);
    end
    if istable(pred)
        pred = table2array(pred);
    end
    rmse = computeRMSE(obs, pred);
    stdv = std(pred);
end

function [rmse] = computeRMSE(obs, pred)
    rmse = sqrt(sum((obs - pred).^2)/height(obs));
end