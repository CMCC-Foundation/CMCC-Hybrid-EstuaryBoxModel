%% Function to train and test a lsboost regression model
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
%  4) k-fold to use in cross-validation

%% Output:
%  Compact structure with the following data:
%  
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
%  4)featuresImportanceTable:
%  Table with features and score which indicates how important is each 
%  feature to train the model. Features have been ordered from the most 
%  important to the least important.
%
%  5) tuningResult:
%  Table with the optimized hyperparameters obtained by auto-tuning
%  procedure

function [results] = lsboost_function(trainingDataset,targetFeatureName,max_objective_evaluations, k)
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

%% Set parameters to be optimized during the auto-tuning procedure
lsboost_settings_optimized = fitrensemble( ...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'OptimizeHyperParameters',...
    {'NumLearningCycles','LearnRate','MinLeafSize','MaxNumSplits'}, ...
    "HyperparameterOptimizationOptions", ...
    struct(...
    "Optimizer", "bayesopt",...
    "AcquisitionFunctionName","expected-improvement-per-second-plus", ...
    "MaxObjectiveEvaluations", max_objective_evaluations,...
    'CVPartition', cross_validation_settings, ...
    "Repartition", false,...
    "UseParallel", true));

%% Save all the optimized hyperparameters
nLearn = lsboost_settings_optimized.ModelParameters.NLearn;
learnRate = lsboost_settings_optimized.ModelParameters.LearnRate;
modelParams = ...
    struct(lsboost_settings_optimized.ModelParameters.LearnerTemplates{1,1});
minLeaf = modelParams.ModelParams.MinLeaf;
maxNSplits = modelParams.ModelParams.MaxSplits;

tuningResult = table('Size', [1 4], 'VariableTypes',...
   {'double','double','double','double'}, 'VariableNames',...
   {'nLearn','minLeaf','learnRate','maxNumSplits'});

tuningResult.nLearn(1) = nLearn;
tuningResult.minLeaf(1) = minLeaf;
tuningResult.learnRate(1) = learnRate;
tuningResult.maxNumSplits(1) = maxNSplits;

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
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));

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