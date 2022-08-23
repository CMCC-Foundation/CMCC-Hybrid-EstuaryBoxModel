%% This function is used to read dataset from an xlsx file and save in into a table.
function [dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes)
opts = spreadsheetImportOptions("NumVariables", nVars);

% Specify sheet and range
opts.DataRange = dataRange;
opts.Sheet = sheetName;

% Specify column names and types
opts.VariableNames = varNames;
opts.VariableTypes = varTypes;

% Specify variable properties
opts = setvaropts(opts, varNames, "EmptyFieldRule", "auto");

% Import the data
dataset = readtable(filepath, opts, "UseExcel", false);
end