__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

from tsa import Logger
from tsa import UVariateTimeSeriesClass
from tsa import print_attributes

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from time import time


class LinearForecaster(UVariateTimeSeriesClass):
    """Univariate time series child class using LinearRegression for forecasting

    Attributes
    ----------
    _fit_intercept: bool
        Whether to fit the intercept yes/no
    _normalize: bool
        Whether to normalize time series data before fitting yes/no
    _copy_X: bool
      If True, X will be copied; else, it may be overwritten.
    _n_jobs: int or None
      The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and
      sufficient large problems. None means 1 unless in a joblib.parallel_backend context.
      -1 means using all processors.

    Methods
    ----------
    assertions()
       Assertion tests, must be overrided
    set_params()
       Sets new parameter values
    get_params_dict()
        Gets parameter values as a dictionary
    ts_fit()
       Fits the auto_arima model to time series
    ts_diagnose()
       Diagnoses the fitted model
    plot_residuals()
       Generates residual plots
    ts_test()
       Evaluates fitted model on the test data, if this one has been generated
    ts_forecast()
       Forecasts time series and plots the results
    plot_forecasts()
       Plots forecasted time-series
    """

    def __init__(self,
                 fit_intercept=True,
                 normalize=False,
                 copy_X=False,
                 n_jobs=None,
                 **kwds):
        """Initializes the object LinearForecaster"""
        self._lin_logger = Logger('linear')

        try:
            super(LinearForecaster, self).__init__(**kwds)
        except TypeError:
            self._lin_logger.exception("Arguments missing...")

        self._fit_intercept = fit_intercept
        self._normalize = normalize
        self._copy_X = copy_X
        self._n_jobs = n_jobs

        self.intercept = None
        self.slope = None

        self._id = 'Linear'

    def __copy__(self):
        """Copies the object"""
        result = super(LinearForecaster, self).__copy__()

        result._fit_intercept = self._fit_intercept
        result._normalize = self._normalize
        result._copy_X = self._copy_X
        result._n_jobs = self._n_jobs
        result.intercept = self.intercept
        result.slope = self.slope
        result._lin_logger = self._lin_logger

        return result

    def set_params(self, p_dict=None, **kwargs):
        """Sets new parameters"""
        params_dict = kwargs
        if p_dict is not None: 
            params_dict = p_dict
        #
        for k, v in params_dict.items():
            if k == 'ts_df':
                self.ts_df = v
            elif k == 'freq':
                self.freq = v
            elif k == 'n_test':
                self.n_test = v
            elif k == 'n_val':
                self.n_val = v
            elif k == 'timeformat':
                self.time_format = v
            elif k == 'fit_intercept':
                self._fit_intercept = v
            elif k == 'normalize':
                self._normalize = v
            elif k == 'copy_X':
                self._copy_X = v
            elif k == 'n_jobs':
                self._n_jobs = v

        return self

    def get_params_dict(self):
        """Gets parameters as dictionary"""
        return {'fit_intercept': self._fit_intercept,
                'normalize': self._normalize,
                'copy_X': self._copy_X,
                'n_jobs': self._n_jobs
                }

    def ts_fit(self, suppress=False):
        """Fit LinearRegression to the time series data.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        self._prepare_fit()
        self.ts_split()

        ts_df = self._train_dt.copy()
        #
        x = np.arange(0, len(ts_df)).reshape(-1, 1)
        y = np.asarray(ts_df['y'])

        # Fit
        self._lin_logger.info("Trying to fit the linear model....")
        # tic
        start = time()
        try:
            if not suppress:
                self._lin_logger.info("...via using parameters")
                print_attributes(self)

            self.model_fit = LinearRegression(fit_intercept=self._fit_intercept,
                                              normalize=self._normalize,
                                              copy_X=self._copy_X,
                                              n_jobs=self._n_jobs).fit(x, y)
            # toc
            self._lin_logger.info("Time elapsed: {} sec.".format(time() - start))
        except (Exception, ValueError):
            self._lin_logger.exception("LinearRegression error...")
        else:
            #
            self._lin_logger.info("Model successfully fitted to the data!")
            if not suppress:
                self._lin_logger.info("R^2: {:f}".format(self.model_fit.score(x, y)))
            #
            self.intercept = self.model_fit.intercept_
            self.slope = self.model_fit.coef_

            # Fitted values
            self._lin_logger.info("Computing fitted values and residuals...")
            self.fittedvalues = pd.Series(self.model_fit.predict(x), index=ts_df.index)

            # Residuals
            super(LinearForecaster, self)._residuals()
            self._lin_logger.info("Done.")
            return self

    def ts_diagnose(self):
        """Diagnoses the model"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._lin_logger.exception("Model has to be fitted first! Please call ts_fit(...)")
            sys.exit("STOP")

        self.plot_residuals()

    def plot_residuals(self):
        """Plot the residuals"""
        fig, axis = super(LinearForecaster, self)._plot_residuals(y=np.asarray(self._train_dt['y']),
                                                                  yhat=np.asarray(self.fittedvalues), _id="Linear")

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        if super(LinearForecaster, self)._check_ts_test() < 0:
            return

        n_forecast = len(self._test_dt)

        self._lin_logger.info("Evaluating the fitted Linear model on the test data...")
        x_future = np.arange(len(self._train_dt), len(self._train_dt) + n_forecast).reshape(-1, 1)
        self.forecast = pd.Series(self.model_fit.predict(x_future), index=self._test_dt.index)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt['y']) - np.asarray(self.forecast),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._lin_logger.info("RMSE on test data: {}".format(self.rmse))
        # plot
        if show_plot:
            self.plot_forecast()
        
        return self            

    def ts_forecast(self, n_forecast, suppress=False):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(LinearForecaster, self)._check_ts_forecast(n_forecast)
        #
        if not suppress:
            self._lin_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._lin_logger.info("Forecasting next " + str(n_forecast) + str(self.freq))
        #
        x_future = np.arange(len(self._train_dt), len(self._train_dt) + n_forecast).reshape(-1, 1)
        future = self.model_fit.predict(x_future)
        idx_future = self._gen_idx_future(n_forecast=n_forecast)
        self.forecast = pd.Series(future, index=idx_future)

        self.residuals_forecast = None
        self.plot_forecast()
        return self

    def plot_forecast(self):
        """Plot forecasted values"""
        fig, axis = super(LinearForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                 yhat=np.asarray(self.fittedvalues),
                                                                 forecast=self.forecast, _id='Linear')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()