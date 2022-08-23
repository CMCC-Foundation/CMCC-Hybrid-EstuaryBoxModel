function [train_set,test_set] = create_training_test_dataset(dataset, test_dataset_size)
%CREATE_TRAINING_TEST_DATASET Split dataset in training set and test set
%   dataset: the original dataset which we want to split in training set
%   and test set
%   test_dataset_size: Fraction or number of observations in the test set 
%   used for holdout validation
    hpartition = cvpartition(height(dataset),'Holdout', test_dataset_size);
    idxTrain = training(hpartition);
    train_set = dataset(idxTrain,:);
    idxTest = test(hpartition);
    test_set = dataset(idxTest,:);
end

