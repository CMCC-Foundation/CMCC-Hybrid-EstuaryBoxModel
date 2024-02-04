function [dataset] = import_dataset(filepath, nVars, dataRange, sheetName, varNames, varTypes)
%IMPORT_DATASET function is used to read dataset 
% from an xlsx file and save in into a table.
%   Input: 
%   1) filepath - path of dataset
%   
%   2) nVars - number of variables in the table
%
%   3) dataRange - start and ending range of excel file
%
%   4) sheetName - name of sheet from which to read the data
%
%   5) varNames - name of variables in the table
%   
%   6) varTypes - datatype of each variables
%
%   Output:
%   1) dataset - the table read from the path

opts = spreadsheetImportOptions("NumVariables", nVars);

% Specify sheet and range
opts.DataRange = dataRange;
opts.Sheet = sheetName;
opts.VariableNamingRule = "preserve";

% Specify column names and types
opts.VariableNames = varNames;
opts.VariableTypes = varTypes;

% Specify variable properties
opts = setvaropts(opts, varNames, "EmptyFieldRule", "auto");

% Import the data
dataset = readtable(filepath, opts, "UseExcel", false);
end