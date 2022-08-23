%% Function to train and test a gam regression model
%% Input:
%  1) trainingDataset: 
%  Table containing the same predictor and response columns as those 
%  imported into the app.
%  
%  2) targetFeatureName: 
%  String with the name of the target feature in the trainingData table.
%  
%  3) max_objective_evaluations:
%  Maximum number of objective functions to be evaluated in the
%  optimization process
%  
% 4) k-fold to use in cross-validation
%% Output:
%  Compact structure with the following data:
%  1) trainedModel:
%  Struct containing the trained regression model. The
%  struct contains various fields with information about the trained
%  model. 
%  trainedModel.predictFcn: A function to make predictions on new data.
%
%  2) validationRMSE: 
%  Double containing the RMSE which measure the performance of the trained
%  model.
%       
%  3) validationPredictions: 
%  Vector with the predected values with respect the observed values in the
%  trainingDataset
%
%  4) tuningResult:
%  Table with the optimized hyperparameters obtained by auto-tuning
%  procedure
function [results] ...
= gam_function(trainingDataset, targetFeatureName, max_objective_evaluations,k)
%% Extract predictors and response
inputTable = trainingDataset;

% Retrive all the features to be used in the training process
predictorNames = inputTable.Properties.VariableNames;
predictorNames(:,(strncmp(predictorNames, targetFeatureName,...
        strlength(targetFeatureName)))) = [];
predictors = inputTable(:, predictorNames);

% Retrive the target feature
response = inputTable(:, targetFeatureName);

%% Set parameters to be optimized during the auto-tuning procedure
rng('default')
gamSettingsOptimized = fitrgam(predictors,response,...
    'OptimizeHyperparameters','auto', ...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName','expected-improvement-plus', ...
    'MaxObjectiveEvaluations', max_objective_evaluations));

%% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(gamSettingsOptimized, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

%% Add additional fields to the result struct
trainedModel.RequiredVariables = predictorNames;
trainedModel.RegressionEnsemble = gamSettingsOptimized;
trainedModel.About = 'This struct is a gam optimized trained model.';
trainedModel.HowToPredict = ...
    sprintf(['To make predictions on a new table, T, use: ' ...
    '\n  yfit = trained_model.predictFcn(T) \n' ...
    '\n \nThe table, T, must contain the variables returned by: ' ...
    '\n  trained_model.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype)' ...
    ' must match the original training data. \nAdditional variables are ignored. ' ...
    '\n \nFor more information, ' ...
    'see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ' ...
    '''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.']);

%% Perform cross-validation with k = 5
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', k);
validationPredictions = kfoldPredict(partitionedModel);
validationRMSE = sqrt(kfoldLoss(partitionedModel));


results = struct('model', trainedModel, 'rmse', validationRMSE,...
    'predictions', validationPredictions, 'hyperparameters', gamSettingsOptimized);
end