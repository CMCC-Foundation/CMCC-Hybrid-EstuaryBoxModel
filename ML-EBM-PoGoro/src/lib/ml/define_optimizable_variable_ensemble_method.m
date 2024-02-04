function [params_settings] = define_optimizable_variable_ensemble_method( predictors, response, ...
    num_learn_cycles, learn_rate, min_leaf_size, max_num_splits, num_var_to_sample)

%% Set parameters to be optimized during the auto-tuning procedure
params_settings = hyperparameters("fitrensemble", predictors, response,'Tree');

% Method optimizaton set to false
params_settings(1).Optimize = false;

% NumLearningCycles 
if num_learn_cycles == 0
    params_settings(2).Optimize = false;
else
    params_settings(2).Optimize = true;
    params_settings(2).Range = num_learn_cycles;
end

%LearnRate
if learn_rate == 0
    params_settings(3).Optimize = false;
else
    params_settings(3).Optimize = true; 
    params_settings(3).Range = learn_rate;
end

%MinLeafSize
if min_leaf_size == 0
    params_settings(4).Optimize = false;
else
    params_settings(4).Optimize = true;
    params_settings(4).Range = min_leaf_size;
end

% MaxNumSplits
if max_num_splits == 0
    params_settings(5).Optimize = false;
else
    params_settings(5).Optimize = true;
    params_settings(5).Range = max_num_splits;
end

% NumVariablesToSample
if num_var_to_sample == 0
    params_settings(6).Optimize = false;
else
    params_settings(6).Optimize = true;
    params_settings(6).Range = num_var_to_sample;
end
end