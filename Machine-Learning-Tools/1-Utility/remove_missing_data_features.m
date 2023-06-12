function [dataset] = remove_missing_data_features(dataset)
%REMOVING_MISSING_DATA_FEATURES Missing values are highlighted with -999
%   This function check each features to find and remove observation in
%   which there are missing values
%   Input: 
%   1) dataset - Table from which removing missing data
%
%   Output:
%   1) dataset - Table with missing data removed
    for i = 2:width(dataset)
        idx = table2array(dataset(:,i)) == -999;
        dataset(idx,:) = [];
    end
end