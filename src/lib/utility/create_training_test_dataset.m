function [train_set,test_set] = create_training_test_dataset(dataset, test_dataset_size)
%CREATE_TRAINING_TEST_DATASET Split dataset in training set and test set
%   Input: 
%   1) dataset - the original dataset which we want to split in training set
%   and test set
%   
%   2) test_dataset_size: Fraction or number of observations in the test set 
%   used for holdout validation
%
%   Output:
%   1) train_set - the training dataset 
%   
%   2) test_set - the test dataset

    rng('shuffle');
    hpartition = cvpartition(height(dataset),'Holdout', test_dataset_size);
    idxTrain = training(hpartition);
    train_set = dataset(idxTrain,:);
    idxTest = test(hpartition);
    test_set = dataset(idxTest,:);
end

