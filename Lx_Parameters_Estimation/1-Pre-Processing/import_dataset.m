%% This function is used to read dataset from an xlsx file and save in into a table.
function [dataset] = import_dataset(filepath)
opts = spreadsheetImportOptions("NumVariables", 7);

% Specify sheet and range
opts.DataRange = "A2:G26";
opts.Sheet = "Lx_obs";

% Specify column names and types
opts.VariableNames = ["DATE","Q_l", "Q_r", "S_l", "Q_tide", "Lx_OBS", "Dataset_Type"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double","double","categorical" ];

% Specify variable properties
opts = setvaropts(opts, ["DATE","Q_l", "Q_r", "S_l", "Q_tide", "Lx_OBS", "Dataset_Type"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "DATE", "InputFormat", "");

% Import the data
dataset = readtable(filepath, opts, "UseExcel", false);
end