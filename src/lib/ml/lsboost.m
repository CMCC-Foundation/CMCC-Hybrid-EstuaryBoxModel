function [results] = lsboost(trainingDataset,targetFeatureName,max_objective_evaluations, k)
%LSBOOST Function to train LSBoost regression model
%   Input:
%   1) trainingDataset: 
%   Table containing the same predictor and response columns as those 
%   imported into the app.
%  
%   2) targetFeatureName: 
%   String with the name of the target feature in the trainingData table.
%  
%   3) max_objective_evaluations:
%   Maximum number of objective functions to be evaluated in the
%   optimization process     
%
%   4) k-fold to use in cross-validation
%
%   Output: results
%   Compact structure with the following data:
%  
%   1) model:
%   Struct containing the trained regression model. The
%   struct contains various fields with information about the trained
%   model. 
%   trainedModel.predictFcn: A function to make predictions on new data.
%
%   2) validation_results: 
%   Structure in which will be store the training performance and the
%   training predictions
%       
%   3) test_results: 
%   Structure in which will be store the test performance and the test
%   predictions
%
%   4)feature_importance:
%   Table with features and score which indicates how important is each 
%   feature to train the model. Features have been ordered from the most 
%   important to the least important.
%
%   5) hyperparameters:
%   Table with the best hyperparameters obtained by hyperparameters
%   optimization

%% Extract predictors and response
inputTable = trainingDataset;

% Retrive all the features to be used in the training process
predictorNames = inputTable.Properties.VariableNames;
predictorNames(:,(strncmp(predictorNames, targetFeatureName,...
        strlength(targetFeatureName)))) = [];
predictors = inputTable(:, predictorNames);

% Retrive the target feature
response = inputTable(:, targetFeatureName);

% Set configuration for k-fold cross validation
cross_validation_settings = cvpartition(height(response),'KFold',k);

params_settings = hyperparameters("fitrensemble", predictors, response,'Tree');
params_settings(5).Optimize = true;
params_settings(6).Optimize = true;

%% Set parameters to be optimized during the auto-tuning procedure
lsboost_settings_optimized = fitrensemble( ...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'OptimizeHyperParameters',...
    params_settings, ...
    "HyperparameterOptimizationOptions", ...
    struct(...
    "Optimizer", "bayesopt",...
    "AcquisitionFunctionName","expected-improvement-per-second-plus", ...
    "MaxObjectiveEvaluations", max_objective_evaluations,...
    'CVPartition', cross_validation_settings, ...
    "Repartition", false,...
    "UseParallel", true));

%% Save all the optimized hyperparameters
modelParams = struct(lsboost_settings_optimized.ModelParameters.LearnerTemplates{1,1});
tuningResult = table('Size', [1 5], 'VariableTypes',...
   {'double','double','double','double', 'double'}, 'VariableNames',...
   {'nLearn','minLeaf','learnRate','maxNumSplits', 'numVarToSample'});

tuningResult.nLearn(1) = lsboost_settings_optimized.ModelParameters.NLearn;
tuningResult.minLeaf(1) = modelParams.ModelParams.MinLeaf;
tuningResult.learnRate(1) = lsboost_settings_optimized.ModelParameters.LearnRate;
tuningResult.maxNumSplits(1) = modelParams.ModelParams.MaxSplits;
tuningResult.numVarToSample(1) = modelParams.ModelParams.NVarToSample;

%% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(lsboost_settings_optimized, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

%% Add additional fields to the result struct
trainedModel.RequiredVariables = trainingDataset.Properties.VariableNames;
trainedModel.RegressionEnsemble = lsboost_settings_optimized;
trainedModel.About = 'This struct is a lsboost optimized trained model.';
trainedModel.HowToPredict = ...
    sprintf(['To make predictions on a new table, T, use: ' ...
    '\n  yfit = trainedModel.predictFcn(T) \n' ...
    '\n \nThe table, T, must contain the variables returned by: ' ...
    '\n  trainedModel.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype)' ...
    ' must match the original training data. \nAdditional variables are ignored. ' ...
    '\n \nFor more information, ' ...
    'see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ' ...
    '''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.']);

%% Perform cross-validation with k = 5
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', k);
validationPredictions = kfoldPredict(partitionedModel);

%% Compute features importance
featureImportance = predictorImportance(lsboost_settings_optimized);
featuresImportanceTable = table('Size', [width(predictorNames) 1], 'VariableTypes',...
    {'double'}, 'VariableNames', {'score'},'RowNames', string(predictorNames'));
    featuresImportanceTable.score = featureImportance';
featuresImportanceTable = sortrows(featuresImportanceTable,'score','descend');

validation_results = struct();
test_results = struct();
validation_results.validation_predictions = validationPredictions;

results = struct('model', trainedModel, ...
    'validation_results', validation_results, ...
    'test_results', test_results,...
    'feature_importance', featuresImportanceTable, ...
    'hyperparameters', tuningResult);
end