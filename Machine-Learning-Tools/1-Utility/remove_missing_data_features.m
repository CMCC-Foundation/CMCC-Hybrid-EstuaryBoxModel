function [dataset] = remove_missing_data_features(dataset)
%REMOVING_MISSING_DATA_FEATURES Missing values are highlighted with -999
%   This function check each features to find and remove observation in
%   which there are missing values
    for i = 2:width(dataset)-1
        idx = table2array(dataset(:,i)) == -999;
        dataset(idx,:) = [];
    end
end