Univariate Time Series Analysis

This package is intended to help you to perform univariate time-series forecasting/analysis. It assumes that you provide univariate time-series data.
You can choose between methods 

- 'lin': linear regression,
- 'auto_arima': ARIMA with grid search and automatic choice of the best model, 
- 'ses': simple exponential smoothing, 
- 'es': exponential smoothing (depending on casted parameters either double or triple), and 
- 'prophet': a more powerful model able to consider holidays, specific seasonalities etc. Ref. to [Prophet](https://facebook.github.io/prophet)

Besindes than these, you can do seasonal decomposition to visually detect anomalies (like spikes) in the data. Refer to ./examples for usage.

The repository structure is as follows:

	time-series-analysis
	|-- tsf_bmw
	|   |-- __init__.py
	|   |-- tools.py
	|   |-- uvtsf.py
    |-- data
    |-- doc
	|-- example-notebooks
    |   |--google_stock.ipynb
    |   |--position.ipynb
    '-- setup.py
    '-- LICENSE
    '-- requirements.txt
    '-- readme.md 

Use pip to install all of a project's dependencies at once by typing pip install -r requirements.txt in your command line.

The package tsf containes the class UVariateTimeSeriesForecaster which you should import as 

	from tsf_bmw.uvtsf import UVariateTimeSeriesForecaster

tools.py offers you some algorithms and wil be subject to development. Right now, moving_linreg can be used. This algorithm performs moving linear regression and computes moving linear slopes over windows of data, the size of which can be casted via a parameter. position.ipynb under example_notebooks demonstrates the application of this algorithm.

Some good Internet resources:

- [ARIMA #1](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)
- [ARIMA #2](https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788)
- [ExponentialSmoothing #1](http://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html#statsmodels.tsa.holtwinters.ExponentialSmoothing.fit)
- [ExponentialSmoothing #2](https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html)
- [ExponentialSmoothing #3](https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/)
- [Prophet](https://facebook.github.io/prophet)

Check also ./doc for further documentation.
