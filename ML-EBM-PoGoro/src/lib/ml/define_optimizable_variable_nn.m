function [params_settings] = define_optimizable_variable_nn(predictors, response, ...
    n_layers, n_units_in_layer, activations, standardize, lambda, layer_weight, layer_bias)
	
%% Set parameters to be optimized during the auto-tuning procedure
params_settings = hyperparameters("fitrnet", predictors, response);

% numLayers
if n_layers == 0
    params_settings(1).Optimize = false;
else
    params_settings(1).Optimize = true;
    params_settings(1).Range = n_layers;
end

% activations
params_settings(2).Optimize = activations;

% standardize
params_settings(3).Optimize = standardize;

% lambda
if lambda == 0
    params_settings(4).Optimize = false;
else
    params_settings(4).Optimize = true;
    params_settings(4).Range = lambda;
end

% Optimize LayerWeightsInitializer
params_settings(5).Optimize = layer_weight;

% Optimize LayerBiasesInitializer
params_settings(6).Optimize = layer_bias;

% Set the number of units for each layer
% Default: 1 - 300
% Min: 1, Max: 400
for ii = 7:7+n_layers(2)-1
    params_settings(ii).Optimize = true;
    params_settings(ii).Range = n_units_in_layer;
    % If we set 5 as max number of layers, then we have to optimize also the
    % layer 4 and 5, and to do that, set to true the 'Optimize' param.

end
end