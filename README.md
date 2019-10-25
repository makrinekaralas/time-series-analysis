# Building Block to Time Series Analysis and Forecasting
# #
This building block is intended to help you to perform univariate (and in the future, some of multivariate) time-series analysis and forecasting. It assumes that you provide time-series data.

The building block is still under the development. Any suggestions for improvement are welcome @ maka.karalashvili@bmw.de.

## Interface

The building block provides you with some ready to use forecasters within an interface that can be easily extended by a new - your own - forecaster.

`UVariateTimeSeriesClass` is the basis class which stores your time series data and provides you with useful methods for time series like resampling, transformation, differencing, decomposition, stationarity tests, ACF and PACF. 

`LinearForecaster`, `ExponentialSmoothingForecaster`, `ARIMAForecaster`, `SARIMAForecaster`, `AutoARIMAForecaster`,   `ProphetForecaster` and `DLMForecaster` inherit from the `UVariateTimeSeriesClass` and extend it each time via the methods for model fitting, model testing and forecasting using respective fitting algorithms.
A try has been made to document all these forecasters within the respective .py files they are defined.

`UVariateTimeSeriesForecaster` is a forecaster that inherits from all forecasters and gives you a powerful interface for univariate time series forecasting. Specifically, here the best model among different models is selected based on the  goodness of models. This latter one is measured via the Root Mean Squared Error (RMSE) over the test data. The list of models can be casted via the parameter 'forecasters'.

`EnsembleForecaster` this forecaster builds the best ensemble of models casted via the parameter 'ensemble' as the list. Firstly, each modell is optimized via running grid search using the provided hyper parameters. These hyper parameters have to be given for each model. If no hyper parameters present for a model, parameters have to be casted, otherwise default values will be used.
Optimized models are then combined - ensembled - in that forecasted values for all possible combinations of models are simply aggregated or left alone. As the aggregation right now mean and meadian are used. The goodness of each such ensemble is measured via the RMSE over the test data. The ensemble with the lowest RMSE is chosen as the best one. 

The inheritance and the hierarchy allows user to work with each type of forecaster via a single interface:

Instantiate a Forecaster &rarr; Apply transformation on time series (optional) &rarr; Fit the Model/Models &rarr; Test and Diagnose the Model/Models &rarr; Validate the Model/Models &rarr; Choose the Best Model &rarr (whenever applies) &rarr; Make Forecast in future

## Forecasters 

Each class provides you via a specific forecaster. Specifically, in case you would know a priori which forecaster is the best for your use-case, you can directly use the respective class. If you are not sure in this matter though, use `UVariateTimeSeriesForecaster` and provide it via a list of fitting algorithms like, e.g. forecasters=['linear', 'auto_arima', 'prophet']; the best model out of these will be chosen for you.

- `LinearForecaster` - a simple linear regression,
- `ARIMAForecaster` - ARIMA,
- `SARIMAForecaster` - seasonal ARIMA, extends `ARIMAForecaster` 
- `ExponentialSmoothingForecaster` - simple (single), double or tripple exponential smoothing,
- `AutoARIMAForecaster` - ARIMA with grid search and automatic choice of the best model, 
- `ProphetForecaster` - a more powerful model able to consider holidays, specific seasonalities etc, [Prophet](https://facebook.github.io/prophet)
- `DLMForecaster` - a state-space model, [DLM](https://pydlm.github.io/)
- `UVariateTimeSeriesForecaster` - a forecaster that chooses the best model out of the provided list (via parameter 'forecasters') based on the goodness of fit (gof) of each model, gof being measured as the RMSE on the test data. Test data must be generated via casting the parameter n_test > 0.
- `EnsembleForecaster` - this forecaster will optimize models via running grid search for each model, these models  provided via a parameter 'ensemble' as a list, using the provided hyper parameters for each model. Both, test and validation data must be generated via casting the parameter n_test > 0 and n_val > 0, respectively. The validation is important here to check how the best ensemble generalizes.

## Model Optimization
Each forecaster except `AutoARIMAForecaster` and `UVariateTimeSeriesForecaster` can be provided a dictionary of hyper parameters via casting 'hyper_params'. In such case, grid search will be automatically run to test all parameter combinations. Finally, the best parameter combination is chosen based on gof measured as RMSE over the test data. In case hyper_params is casted, test data must be generated via casting the parameter n_test > 0.

Note, that `AutoARIMAForecaster` already conducts grid search for you, whereas `UVariateTimeSeriesForecaster` with hyper_params results in `EnsembleForecaster`. So, if you want to optimize many models just use `EnsembleForecaster`.

## Integrating a new Forecaster

You can implement and integrate your own forecaster. For this, you will need to create a class that inherits from `UVariateTimeSeriesClass` and implement ts_fit() method for it. We recommend you to use one of already provided forecasters as a template and to replace its ts _ fit() method. Possibly, you will also need to adapt ts _ test() and ts _ forecast() methods. For any questions do not hasitate to reach us out.  

To integrate your own forecaster as a possible model candidate within `UVariateTimeSeriesForecaster`, you will need to adapt ts _ fit() method of `UVariateTimeSeriesForecaster` and extend the attribute dict_models via your own forecaster. Additionally, `UVariateTimeSeriesForecaster` has to inherit from your new forecaster.

## Repository structure

The repository structure is as follows:

	time-series-analysis
	|-- tsa
	|   |-- __init__.py
	|   |-- uvts.py
	|   |-- linear.py
	|   |-- arima.py
	|   |-- sarima.py
	|   |-- exp_smoothing.py
	|   |-- auto_arima.py
	|   |-- prophet.py
	|   |-- dlm.py
	|   |-- ensemble.py
	|   |-- uvtsf.py
	|   |-- logger.py
	|   |-- tools.py
    |-- data
	|-- use-cases
	|-- py_scripts
    |   |--run_auto_arima.py
    '-- setup.py
    '-- LICENSE
    '-- requirements.txt
    '-- readme.md 

Use pip to install all of a project's dependencies at once by typing pip install -r requirements.txt in your command line.

### tools.py
tools.py offers you some moving window algorithms and is the subject to development. Please do not use it now.
moving_ linreg is a simple algorithm that performs simply moving linear regression and computes moving linear slopes over windows of data, the size and moving step of which can be casted via a parameter. position.ipynb under example_notebooks demonstrates the application of this algorithm.


## References and Memo
### References
- [ARIMA #1](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)
- [ARIMA #2](https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788)
- [ExponentialSmoothing #1](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html#statsmodels.tsa.holtwinters.ExponentialSmoothing.fit)
- [ExponentialSmoothing #2](https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html)
- [ExponentialSmoothing #3](https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/)
- [Prophet](https://facebook.github.io/prophet)
- [DLM](https://pydlm.github.io/)
- [Python classes](https://rhettinger.wordpress.com/2011/05/26/super-considered-super/)
### Memo
Recall, the Box-Cox transform is given by:
y = (x**lmbda - 1) / lmbda,  for lmbda > 0
    log(x),                  for lmbda = 0

