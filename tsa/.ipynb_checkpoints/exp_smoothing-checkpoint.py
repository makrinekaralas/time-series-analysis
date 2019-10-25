#!/usr/bin/env python.
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
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from time import time


class ExponentialSmoothingForecaster(UVariateTimeSeriesClass):
    """Univariate time series child class using simple, double or triple exponential smoothing for forecasting

    Attributes
    ----------
    ref. to e.g., https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/
    _optimized: bool
        Whether to optimize smoothing coefficients
    _smoothing_level: float
       (alpha): the smoothing coefficient for the level
    _es_trend: str
        The type of trend component, as either “add” for additive or “mul” for multiplicative.
        Modeling the trend can be disabled by setting it to None
    _damped: bool
        Whether or not the trend component should be damped, either True or False
    _es_seasonal: str
        The type of seasonal component, as either “add” for additive or “mul” for multiplicative.
        Modeling the seasonal component can be disabled by setting it to None
    _seasonal_periods: int
         The number of time steps in a seasonal period, e.g. 12 for 12 months in a yearly seasonal structure
    _smoothing_slope: float
       (beta): the smoothing coefficient for the trend
    _smoothing_seasonal: float
       (gamma): the smoothing coefficient for the seasonal component
    _damping_slope: float
       (phi): the coefficient for the damped trend
    _use_boxcox: {True, False, ‘log’, float}
       Should the Box-Cox transform be applied to the data first? If ‘log’ then apply the log.
       If float then use lambda equal to float
    _remove_bias: bool
       Remove bias from forecast values and fitted values by enforcing that the average residual is equal to zero.
    _use_brute: bool
       Search for good starting values using a brute force (grid) optimizer.
       If False, a naive set of starting values is used.
    _expsm_logger: Logger
       The logger for logging

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
                 smoothing_level=None,
                 optimized=False,
                 damped=False,
                 smoothing_slope=None,
                 smoothing_seasonal=None,
                 damping_slope=None,
                 use_boxcox=False,
                 remove_bias=False,
                 use_brute=False,
                 **kwds):
        """Initializes the object ExponentialSmoothingForecaster"""

        self._expsm_logger = Logger("ExpSmoothing")
        self._es_trend = None
        self._es_seasonal = None

        try:
            super(ExponentialSmoothingForecaster, self).__init__(**kwds)
        except TypeError:
            self._expsm_logger.exception("Arguments missing...")

        self._init_trend()
        self._init_seasonal()

        self._smoothing_level = smoothing_level
        self._optimized = optimized
        self._damped = damped
        self._smoothing_slope = smoothing_slope
        self._smoothing_seasonal = smoothing_seasonal
        self._damping_slope = damping_slope
        self._use_boxcox = use_boxcox
        self._remove_bias = remove_bias
        self._use_brute = use_brute
        
        self.assertions()
        
        self._id = 'ExponentialSmoothing'

    def _init_trend(self):
        if self._trend is None or self._trend == 'constant':
            self._es_trend = None
        elif self._trend in ['linear', 'constant linear']:
            # self._expsm_logger.warning("The trend " + self(self._trend) + " not supported in Exponential Smoothing! "
            #                                                              "Assuming additive trend")
            self._es_trend = 'add'
        else:
            self._es_trend = self._trend

    def _init_seasonal(self):
        if isinstance(self._seasonal, bool):
            if self._seasonal:
                # self._expsm_logger.warning("Assuming additive seasonal component in Exponential Smoothing")
                self._es_seasonal = 'add'
            else:
                self._es_seasonal = None
        else:
            self._es_seasonal = self._seasonal

    def __copy__(self):
        """Copies the object"""
        result = super(ExponentialSmoothingForecaster, self).__copy__()

        result._smoothing_level = self._smoothing_level
        result._optimized = self._optimized
        result._es_trend = self._es_trend
        result._es_seasonal = self._es_seasonal
        result._damped = self._damped
        result._smoothing_slope = self._smoothing_slope
        result._smoothing_seasonal = self._smoothing_seasonal
        result._damping_slope = self._damping_slope
        result._use_boxcox = self._use_boxcox
        result._remove_bias = self._remove_bias
        result._use_brute = self._use_brute
        result._expsm_logger = self._expsm_logger

        return result

    def assertions(self):
        try:
            assert self._es_trend is None or self._es_trend in ['add', 'mul', 'additive', 'multiplicative']
        except AssertionError:
            self._expsm_logger.exception("Assertion Error, trend must be in ['add','mul',"
                                         "'additive','multiplicative']")
            sys.exit("STOP")
        try:
            assert self._es_seasonal is None or isinstance(self._es_seasonal, str) and self._es_seasonal in ['add',
                                                                                                             'mul',
                                                                                                             'additive',
                                                                                                             'multiplicative']
        except AssertionError:
            self._expsm_logger.exception("Assertion Error, seasonal must be in ['add','mul',"
                                         "'additive','multiplicative']")
            sys.exit("STOP")

    def set_params(self, p_dict=None, **kwargs):
        """Sets new parameters"""
        params_dict = kwargs
        if p_dict is not None:
            params_dict = p_dict
        #
        for k, v in params_dict.items():
            if k == 'smoothing_level':
                self._smoothing_level = v
            elif k == 'optimized':
                self._optimized = v
            elif k == 'trend':
                self._es_trend = v
            elif k == 'seasonal':
                self._es_seasonal = v
            elif k == 'seasonal_periods':
                self._seasonal_periods = v
            elif k == 'damped':
                self._damped = v
            elif k == 'smoothing_slope':
                self._smoothing_slope = v
            elif k == 'smoothing_seasonal':
                self._smoothing_seasonal = v
            elif k == 'damping_slope':
                self._damping_slope = v
            elif k == 'use_boxcox':
                self._use_boxcox = v
            elif k == 'remove_bias':
                self._remove_bias = v
            elif k == 'use_brute':
                self._use_brute = v
        self.assertions()

        return self

    def get_params_dict(self):
        """Gets parameters as dictionary"""
        return {'smoothing_level': self._smoothing_level,
                'optimized': self._optimized,
                'trend': self._es_trend,
                'seasonal': self._es_seasonal,
                'seasonal_periods': self._seasonal_periods,
                'damped': self._damped,
                'smoothing_slope': self._smoothing_slope,
                'smoothing_seasonal': self._smoothing_seasonal,
                'damping_slope': self._damping_slope,
                'use_boxcox': self._use_boxcox,
                'remove_bias': self._remove_bias,
                'use_brute': self._use_brute
                }

    def ts_fit(self, suppress=False):
        """Fit Exponential Smoothing to the time series data.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        self._prepare_fit()
        self.ts_split()
        self._init_trend()
        self._init_seasonal()

        ts_df = self._train_dt.copy()

        # Fit
        print("Trying to fit the exponential smoothing model....")
        # tic
        start = time()
        try:
            if not suppress:
                self._expsm_logger.info("...via using parameters\n")
                print_attributes(self)
            #
            self.model_fit = ExponentialSmoothing(ts_df,
                                                  freq=self.freq,
                                                  trend=self._es_trend,
                                                  seasonal=self._es_seasonal,
                                                  seasonal_periods=self._seasonal_periods,
                                                  damped=self._damped).fit(smoothing_level=self._smoothing_level,
                                                                           smoothing_slope=self._smoothing_slope,
                                                                           smoothing_seasonal=self._smoothing_seasonal,
                                                                           damping_slope=self._damping_slope,
                                                                           optimized=self._optimized,
                                                                           use_boxcox=self._use_boxcox,
                                                                           remove_bias=self._remove_bias)
        # toc
            self._expsm_logger.info("Time elapsed: {} sec.".format(time() - start))
        except (Exception, ValueError):
            self._expsm_logger.exception("Exponential Smoothing error...")
        else:
            #
            self._expsm_logger.info("Model successfully fitted to the data!")

            # Fitted values
            self._expsm_logger.info("Computing fitted values and residuals...")
            self.fittedvalues = self.model_fit.fittedvalues

            # Residuals
            super(ExponentialSmoothingForecaster, self)._residuals()
            self._expsm_logger.info("Done.")

            return self

    def ts_diagnose(self):
        """Diagnose the model"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._expsm_logger.exception("Model has to be fitted first! Please call ts_fit(...)")
            sys.exit("STOP")

        self.plot_residuals()

    def plot_residuals(self):
        """Plot the residuals"""
        fig, axis = super(ExponentialSmoothingForecaster, self)._plot_residuals(y=np.asarray(self._train_dt['y']),
                                                                                yhat=np.asarray(self.fittedvalues),
                                                                                _id="Exponential Smoothing")

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        
        if super(ExponentialSmoothingForecaster, self)._check_ts_test() < 0:
            return

        n_forecast = len(self._test_dt)

        self._expsm_logger.info("Evaluating the fitted model on the test data...")
        self.forecast = self.model_fit.forecast(n_forecast)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt['y']) - np.asarray(self.forecast),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._expsm_logger.info("RMSE on test data: {}".format(self.rmse))
        # plot
        if show_plot:
            self.plot_forecast()

    def ts_forecast(self, n_forecast, suppress):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(ExponentialSmoothingForecaster, self)._check_ts_forecast(n_forecast)
        #
        self._expsm_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._expsm_logger.info("Forecasting next " + str(n_forecast) + str(self.freq))
        #
        self.forecast = self.model_fit.forecast(n_forecast)

        self.residuals_forecast = None
        # self.plot_forecast()
        return self

    def plot_forecast(self):
        """Plot forecasted values"""
        fig, axis = super(ExponentialSmoothingForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                               yhat=np.asarray(self.fittedvalues),
                                                                               forecast=self.forecast,
                                                                               _id='Exponential Smoothing')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()
