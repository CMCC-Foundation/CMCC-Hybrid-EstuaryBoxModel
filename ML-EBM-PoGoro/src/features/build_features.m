clc
%clear
close all

addpath(genpath("..\..\data\"));
addpath(genpath("..\lib"));

%% Read the raw dataset
sul_dataset_read = import_dataset("raw\raw-dataset-sul.xlsx", 10, "A2:J1462", "Sheet1", ...
    ["ID","Date","Qriver", "Qll", "Qtidef", "Sll", "Socean", "Sul", "Sul_EBM", "Dataset"], ...
    ["categorical","datetime", "double", "double", "double", "double","double", "double", "double", "categorical"]);
sul_dataset = sul_dataset_read;

%% Remove missed data
sul_dataset = sul_dataset(not(sul_dataset.Dataset == "Missed"),:);

%% Add feature: Season
sul_dataset = split_date_in_season(sul_dataset, unique(year(sul_dataset.Date)));

%%%% etc... 

%% Save the dataset
writetable(sul_dataset, "..\..\data\processed\input-features-sul.xlsx");