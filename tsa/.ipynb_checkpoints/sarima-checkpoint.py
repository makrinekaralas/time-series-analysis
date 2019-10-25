__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

from tsa import Logger

from tsa import ARIMAForecaster
from tsa import print_attributes

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from time import time


class SARIMAForecaster(ARIMAForecaster):
    """Univariate time series child class for forecasting using SARIMA

    Attributes
    ----------
    _order: tuple
       a tuple of p, d, q
    _s_order: tuple
        A tuple of seasonal components (P, D, Q, lag)
    _sarima_logger: Logger
        The logger for logging
    _sarima_trend: str
        A parameter for controlling a model of the deterministic trend as one of ‘n’,’c’,’t’,’ct’ for no trend,
        constant, linear, and constant with linear trend, respectively.

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
                 s_order=(1, 0, 1, 1),
                 **kwds):
        """Initializes the object SARIMAForecaster"""
        self._sarima_logger = Logger("SARIMA")

        self._s_order = s_order
        self._sarima_trend = ''

        try:
            super(SARIMAForecaster, self).__init__(**kwds)
        except TypeError as e:
            self._sarima_logger.exception("Arguments missing...{}".format(e))

        self._model = None
        self._init_trend()
        self.assertions()

        self._id = 'SARIMA'

    def _init_trend(self):
        if self._trend == 'constant':
            self._sarima_trend = 'c'
        elif self._trend is None:
            self._srima_trend = 'n'
        elif self._trend == 'linear':
            self._sarima_trend = 't'
        elif self._trend == 'constant linear':
            self._sarima_trend = 'ct'
        elif self._trend in ['additive', 'add']:
            # self._sarima_logger.warningg("The trend " + str(self._trend) + " is not supported by SARIMA! "
            #                                                               "Assuming linear trend")
            self._sarima_trend = 't'
        elif self._trend in ['multiplicative', 'mul']:
            # self._sarima_logger.warning(
            #    "The trend " + str(self._trend) + " is not supported by ARIMA! Assuming linear trend")
            self._sarima_trend = 't'

    def assertions(self):
        try:
            assert isinstance(self._s_order, tuple)
        except AssertionError:
            self._sarima_logger.exception("Assertion exception occurred, tuple expected")
            sys.exit("STOP")
        try:
            assert self._sarima_trend is None or self._sarima_trend in ['n', 'c', 't', 'ct']
        except AssertionError:
            self._sarima_logger.exception("Assertion Error, trend must be in ['n', 'c', 't', 'ct']")
            sys.exit("STOP")
        try:
            assert isinstance(self._seasonal, bool)
        except AssertionError:
            self._sarima_logger.exception("Assertion Error, seasonal must be boolean True/False in SARIMA")
            sys.exit("STOP")

    def __copy__(self):
        """Copies the object"""
        result = super(SARIMAForecaster, self).__copy__()

        result._s_order = self._s_order
        result._sarima_trend = self._sarima_trend
        result._sarima_logger = self._sarima_logger
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
            elif k == 's_order':
                self._s_order = v
            elif k == 'order':
                self._order = v
            elif k == 'test':
                self._test = v
            elif k == 'trend':
                self._sarima_trend = v
        self.assertions()

        return self

    def get_params_dict(self):
        """Gets parameters as a dictionary"""
        return {'order': self._order,
                'test': self._test,
                'trend': self._sarima_trend,
                's_order': self._s_order,
                }

    def ts_fit(self, suppress=False):
        """Fit Seasonal ARIMA to the time series data.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        self._prepare_fit()
        self.ts_split()
        self._init_trend()

        ts_df = self._train_dt.copy()

        # Fit
        self._sarima_logger.info("Trying to fit the sarima model....")
        # tic
        start = time()
        try:
            if not suppress:
                self._sarima_logger.info("...via using parameters\n")
                print_attributes(self)

            self._model = SARIMAX(ts_df['y'], order=self._order,
                                  seasonal_order=self._s_order, trend=self._sarima_trend,
                                  enforce_stationarity=False, enforce_invertibility=False,
                                  freq=self.freq)
            self.model_fit = self._model.fit(disp=1)
        except (Exception, ValueError):
            self._sarima_logger.exception("Exception occurred in the fit...")
            self._sarima_logger.error("Please try other parameters!")
            self.model_fit = None

        else:
            # toc
            self._sarima_logger.info("Time elapsed: {} sec.".format(time() - start))
            self._sarima_logger.info("Model successfully fitted to the data!")
            if not suppress:
                self._sarima_logger.info("The model summary: " + str(self.model_fit.summary()))

            # Fitted values
            self._sarima_logger.info("Computing fitted values and residuals...")
            self.fittedvalues = self.model_fit.fittedvalues
            # prolong: for some reason this package returns fitted values this way
            if len(self.fittedvalues) != len(self._train_dt):
                self.fittedvalues = pd.DataFrame(
                    index=pd.date_range(ts_df.index[0], ts_df.index[len(ts_df) - 1],
                                        freq=self.freq),
                    columns=['dummy']).join(pd.DataFrame(self.fittedvalues)).drop(['dummy'], axis=1)
                self.fittedvalues = self.fittedvalues.reset_index()
                self.fittedvalues.columns = self._ts_df_cols
                self.fittedvalues.set_index('ds', inplace=True)
                self.fittedvalues.y = self.fittedvalues.y.fillna(method='bfill')

            #  Residuals
            super(SARIMAForecaster, self)._residuals()
            self._sarima_logger.info("Done.")
            return self

    def plot_residuals(self):
        """Plot the residuals"""
        fig, axis = super(SARIMAForecaster, self)._plot_residuals(y=np.asarray(self._train_dt['y']),
                                                                  yhat=np.asarray(self.fittedvalues).flatten(),
                                                                  _id="SARIMA")

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        if super(SARIMAForecaster, self)._check_ts_test() < 0:
            return

        n_forecast = len(self._test_dt)

        if self._mode == 'test':
            self._sarima_logger.info("Evaluating the fitted SARIMA model on the test data...")
        elif self._mode == 'test and validate':
            self._sarima_logger.info("Evaluating the fitted SARIMA model on the test and validation data...")

        future = self.model_fit.predict(start=len(self._train_dt.index),
                                        end=len(self._train_dt.index) + n_forecast - 1, dynamic=True)

        self.forecast = pd.Series(future, index=self._test_dt.index)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt.y) - np.asarray(self.forecast),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._sarima_logger.info("RMSE on test data: {}".format(self.rmse))

        # plot
        if show_plot:
            self.plot_forecast()
        
        return self

    def ts_forecast(self, n_forecast, suppress=False):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(SARIMAForecaster, self)._check_ts_forecast(n_forecast)
        #
        self._sarima_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._sarima_logger.info("Forecasting next " + str(n_forecast) + str(self.freq))
        #
        future = self.model_fit.predict(start=len(self._train_dt.index),
                                        end=len(self._train_dt.index) + (n_forecast-1), dynamic=True)
        idx_future = self._gen_idx_future(n_forecast=n_forecast)
        self.forecast = pd.Series(future, index=idx_future)
        #self.forecast = future

        self.residuals_forecast = None
        self.plot_forecast()
        
        return self

    def plot_forecast(self):
        """Plot forecasted values"""
        fig, axis = super(SARIMAForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                 yhat=np.asarray(self.fittedvalues).flatten(),
                                                                 forecast=self.forecast, _id='SARIMA')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()
