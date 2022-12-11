function [results] = exponential_fit_function(xTrain, yTrain, xTest, yTest)

%% Prepare data to fit
[xTrain, yTrain] = prepareCurveData(xTrain, yTrain);

%% Set up fittype and options.
ft = fittype('exp1');
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Normalize = 'on';
opts.Robust = 'Bisquare';
opts.StartPoint = [10.528778557773 -0.605514325290387];

%% Fit model to train data.
[fitresult, validation_results] = fit( xTrain, yTrain, ft, opts );

%% Compare against test data.
[xTest, yTest] = prepareCurveData(xTest, yTest);
yTestPredicted = fitresult( xTest );
residual = yTest - yTestPredicted;

nNaN = nnz( isnan( residual ) );
residual(isnan( residual )) = [];
sse = norm( residual )^2;
rmse = sqrt( sse/length( residual ) );
rsquare = 1 - (sum((residual).^2)/sum((yTest - mean(yTest)).^2));

fprintf( 'Goodness-of-validation for ''%s'' fit:\n', 'untitled fit 1' );
fprintf( '    SSE : %f\n', sse );
fprintf( '    RMSE : %f\n', rmse );
fprintf( '    R-Square : %f\n', rsquare );
fprintf( '    %i points outside domain of data.\n', nNaN );

%% Save results
test_results = struct();
test_results.test_predictions = yTestPredicted;


results = struct('model', fitresult, ...
    'validation_results', validation_results, ...
    'test_results', test_results,...
    'hyperparameters_1', ft,...
    'hyperparameters_2', opts);


%% Plot fit with data.
figure();
h = plot( fitresult, xTrain, yTrain );
% Add validation data to plot.
hold on
h(end+1) = plot( xTest, yTest, 'ko', 'MarkerFaceColor', 'w' );
hold off
legend( h, 'Training observations', 'Model', 'Test observations', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'Qriver', 'Interpreter', 'none' );
ylabel( 'SalinityObs', 'Interpreter', 'none' );
grid on