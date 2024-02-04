function [results, validation_predictions] = ensemble_method( ...
    training_data, predictor_names, target_name, max_obj_eval, k, ml_method, params_settings)
%ENSEMBLE_METHOD Function to train an ensemble regression model
%   Input:
%   1) training_dataset: Table containing the same predictor and response columns
%   2) predictor_names: The features name used to train the model in training_dataset
%   3) target_name: The name of the target feature in training_dataset
%   4) max_objective_evaluations: Maximum number of objective functions to be evaluated in the
%   optimization process
%   5) k: number of folds used in k-fold cross-validation
%   6) ml_method: The ensemble method selected (Bag or LSBoost)
%   7) params_settings: optimizableVariable object with the hyperparameters 
%   selected, with their range for the tuning procedure
%
%   Output: 
%   1) results: Compact structure with the following data:
%       1.1) model: Struct containing the trained regression model. The
%       struct contains various fields with information about the trained
%       model. trainedModel.predictFcn: A function to make predictions on new data.
%       1.2) hyperparameters: table with the optimized hyperparameters
%   2) validation_predictions: predictions on the trained dataset

%% Extract predictors and response
predictors = training_data(:, predictor_names);
response = training_data(:, target_name);

%% Set configuration for k-fold cross validation
cross_validation_settings = cvpartition(height(response),'KFold',k);

%% Optimize hyperparameters
t = templateTree('Surrogate','on');
ensemble_settings_optimized = fitrensemble( ...
    predictors, response, 'Learners', t, 'Method', ml_method, ...
    'OptimizeHyperParameters', params_settings, ...
    'HyperparameterOptimizationOptions', ...
    struct(...
    'Optimizer', 'bayesopt',...
    'AcquisitionFunctionName','expected-improvement-per-second-plus', ...
    'CVPartition', cross_validation_settings, ...
    'MaxObjectiveEvaluations', max_obj_eval,...
    'UseParallel', true, 'Verbose', 2));

ensemble_settings_optimized = regularize(ensemble_settings_optimized, "Verbose", 1);

%RF
%% Save all the optimized hyperparameters
model_params = struct(ensemble_settings_optimized.ModelParameters.LearnerTemplates{1,1});
tuning_result = table();
tuning_result.nLearn = ensemble_settings_optimized.ModelParameters.NLearn;
tuning_result.minLeaf = model_params.ModelParams.MinLeaf;
tuning_result.maxSplits = model_params.ModelParams.MaxSplits;
tuning_result.nVarToSample = model_params.ModelParams.NVarToSample;

if ml_method == "LSBoost"
    tuning_result.learnRate = ensemble_settings_optimized.ModelParameters.LearnRate;
end

%% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictor_names);
ensemblePredictFcn = @(x) predict(ensemble_settings_optimized, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

%% Add additional fields to the result struct
trainedModel.RequiredVariables = predictor_names;
trainedModel.RegressionEnsemble = ensemble_settings_optimized;
trainedModel.About = "This struct is a " + ml_method + " optimized trained model.";
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
partitioned_model = crossval(trainedModel.RegressionEnsemble, 'KFold', k);
validation_predictions = kfoldPredict(partitioned_model);

%% Compute features importance
feature_importance = predictorImportance(ensemble_settings_optimized);
features_importance_table = table('Size', [width(predictor_names) 1], 'VariableTypes',...
    {'double'}, 'VariableNames', {'score'},'RowNames', string(predictor_names'));
    features_importance_table.score = feature_importance';
features_importance_table = sortrows(features_importance_table,'score','descend');

results = struct('model', trainedModel, ...
    'feature_importance', features_importance_table, ...
    'hyperparameters', tuning_result);
end