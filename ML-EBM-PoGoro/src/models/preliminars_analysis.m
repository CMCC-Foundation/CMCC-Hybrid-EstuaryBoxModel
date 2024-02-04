clc
clear
close all
addpath(genpath("..\..\data\"));
addpath(genpath("..\lib"));

%% Read the processed dataset
sul_dataset = import_dataset("processed\input-features-sul.xlsx", 11, "A2:K1202", "Sheet1", ...
    ["ID","Date","Qriver", "Qll", "Qtidef", "Sll", "Socean", "Sul", "Sul_EBM", "Dataset", "Season"], ...
    ["categorical","datetime", "double", "double", "double", "double","double", "double", "double", "categorical", "categorical"]);

sul_dataset.Year = year(sul_dataset.Date);
sul_dataset.Year = categorical(sul_dataset.Year);

%% Create boxplot and perform statistical test by year
% Create boxplot
groupcounts(sul_dataset, "Year")
figure();
plot_boxchart(sul_dataset.Sul, sul_dataset.Year, "Year");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-year-boxplot.png");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-year-boxplot.fig");

% Run statistical test 
[p,tbl,stats] = kruskalwallis(sul_dataset.Sul, sul_dataset.Year); % Confident interval 1%
[c,m,h,gnames] = multcompare(stats,"Alpha", 0.01, "CriticalValueType","dunn-sidak");
tbl_results = array2table(c, "VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"])

%% Create seasonal salinity boxplot year by year
figure();
tiledlayout(2,2);

nexttile
plot_boxchart(sul_dataset.Sul(sul_dataset.Year == "2016"), ...
    sul_dataset.Season(sul_dataset.Year == "2016"), "2016");
nexttile
plot_boxchart(sul_dataset.Sul(sul_dataset.Year == "2017"), ...
    sul_dataset.Season(sul_dataset.Year == "2017"), "2017");
nexttile
plot_boxchart(sul_dataset.Sul(sul_dataset.Year == "2018"), ...
    sul_dataset.Season(sul_dataset.Year == "2018"), "2018");
nexttile
plot_boxchart(sul_dataset.Sul(sul_dataset.Year == "2019"), ...
    sul_dataset.Season(sul_dataset.Year == "2019"), "2019");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-season-by-year-boxplot.png");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-season-by-year-boxplot.fig");

%% Create boxplot and perform statistical test by season
% Create boxplot
groupcounts(sul_dataset, "Season")
figure();
plot_boxchart(sul_dataset.Sul, sul_dataset.Season, "Season");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-season-boxplot.png");
%saveas(gcf,"..\..\reports\figure\Statistical-Plot\sul-season-boxplot.fig");

% Run statistical test
[p,tbl,stats] = kruskalwallis(sul_dataset.Sul, sul_dataset.Season); % Confident interval 1%
[c,m,h,gnames] = multcompare(stats, "Alpha", 0.01, "CriticalValueType","dunn-sidak");
tbl_results = array2table(c, "VariableNames", ...
    ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"])


%% Fucntion to plot boxchart
function plot_boxchart(y, group, xlab)
boxchart(group, y);
xlabel(xlab);
ylabel("Sul (psu)");
ylim([-0.1, 30]);
end