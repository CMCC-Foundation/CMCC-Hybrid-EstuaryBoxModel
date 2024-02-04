function [results, validationPredictions] = neural_network( ...
    training_data, predictor_names, target_name, max_obj_eval, k, params_settings)
%NEURAL_NETWORK Function to train a neural network regression model
%   Input:
%   1) training_dataset: Table containing the same predictor and response columns
%   2) predictor_names: The features name used to train the model in training_dataset
%   3) target_name: The name of the target feature in training_dataset
%   4) max_objective_evaluations: Maximum number of objective functions to be evaluated in the
%   optimization process
%   5) k: number of folds used in k-fold cross-validation
%   6) params_settings: optimizableVariable object with the hyperparameters 
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

% Set configuration for k-fold cross validation
cross_validation_settings = cvpartition(height(response),'KFold',k);

%% Train a regression model
nn_settings_optimized = fitrnet(...
    predictors, response, ...
    'OptimizeHyperparameters', params_settings, ...
    'HyperparameterOptimizationOptions', ...
    struct(...
    'Optimizer', 'bayesopt',...
    'AcquisitionFunctionName','expected-improvement-per-second-plus', ...
    'MaxObjectiveEvaluations', max_obj_eval,...
    'CVPartition', cross_validation_settings, ...
    'UseParallel', true));

%% Save all the optimized hyperparameters
tuning_result = table();
tuning_result.LayerSizes = nn_settings_optimized.ModelParameters.LayerSizes;
tuning_result.Activations = nn_settings_optimized.ModelParameters.Activations;
tuning_result.LayerWeightsInitializer = nn_settings_optimized.ModelParameters.LayerWeightsInitializer;
tuning_result.LayerBiasesInitializer = nn_settings_optimized.ModelParameters.LayerBiasesInitializer;
tuning_result.Lambda = nn_settings_optimized.ModelParameters.Lambda;
tuning_result.Standardize = nn_settings_optimized.ModelParameters.StandardizeData;

%% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictor_names);
neuralNetworkPredictFcn = @(x) predict(nn_settings_optimized, x);
trainedModel.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

%% Add additional fields to the result struct
trainedModel.RequiredVariables = predictor_names;
trainedModel.RegressionNeuralNetwork = nn_settings_optimized;
trainedModel.About = 'This struct is a neural network optimized trained model.';
trainedModel.HowToPredict = ...
    sprintf(['To make predictions on a new table, T, use: ' ...
    '\n  yfit = trained_model.predictFcn(T) \n' ...
    '\n \nThe table, T, must contain the variables returned by: ' ...
    '\n  trained_model.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype)' ...
    ' must match the original training data. \nAdditional variables are ignored. ' ...
    '\n \nFor more information, ' ...
    'see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ' ...
    '''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.']);

%% Perform cross-validation
partitioned_model = crossval(trainedModel.RegressionNeuralNetwork, 'KFold', k);
validationPredictions = kfoldPredict(partitioned_model);
results = struct('model', trainedModel, ...
    'hyperparameters', tuning_result);
end