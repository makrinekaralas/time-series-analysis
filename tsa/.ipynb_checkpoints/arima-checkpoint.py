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
from statsmodels.tsa.arima_model import ARIMA
from time import time


class ARIMAForecaster(UVariateTimeSeriesClass):
    """Univariate time series child class for forecasting using ARIMA

    Attributes
    ----------
    _order: tuple
       a tuple of p, d, q
    _arima_trend: str
        A parameter for controlling a model of the deterministic trend as one of ‘nc’ or ’c’.
        ‘c’ includes constant trend, ‘nc’ no constant for trend.
    _arima_logger: Logger
       the logger

    Methods
    ----------
    assertions()
       Assertion tests, must be overrided
    set_params()
       Sets new parameter values
    get_params_dict()
        Gets parameter values as a dictionary
    ts_fit()
       Fits the ARIMA model to time series
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
                 order=(1, 0, 1),
                 **kwds):
        """Initializes the object ARIMAForecaster"""
        self._arima_logger = Logger("ARIMA")

        self._order = order
        self._arima_trend = ''

        try:
            super(ARIMAForecaster, self).__init__(**kwds)
        except TypeError as e:
            self._arima_logger.exception("Arguments missing...{}".format(e))

        self._model = None

        ARIMAForecaster._init_trend(self)

        self._ar_coef = None
        self._ma_coef = None

        ARIMAForecaster.assertions(self)

        self._id = 'ARIMA'

    def _init_trend(self):
        if self._trend == 'constant':
            self._arima_trend = 'c'
        elif self._trend is None:
            self._arima_trend = 'nc'
        elif self._trend in ['linear', 'constant linear', 'additive', 'add', 'multiplicative', 'mul']:
            # self._arima_logger.warning("The trend " + str(self._trend) +
            #                           " is not supported by ARIMA! Assuming constant trend")
            self._arima_trend = 'c'

    def __copy__(self):
        """Copies the object"""
        result = super(ARIMAForecaster, self).__copy__()

        result._order = self._order
        result._arima_trend = self._arima_trend

        result._arima_logger = self._arima_logger
        return result

    def assertions(self):
        try:
            assert isinstance(self._order, tuple)
        except AssertionError:
            self._arima_logger.exception("Assertion exception occurred, tuple expected")
            sys.exit("STOP")
        try:
            assert self._arima_trend in ['c', 'nc'] or self._arima_trend is None
        except AssertionError:
            self._arima_logger.exception("Assertion Error, trend must be in ['c', 'nc']")
            sys.exit("STOP")

    def set_params(self, p_dict=None, **kwargs):
        """Sets new parameter values"""

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
            elif k == 'time_format':
                self.time_format = v
            elif k == 'order':
                self._order = v
            elif k == 'trend':
                self._arima_trend = v
        self.assertions()

        return self

    def get_params_dict(self):
        """Gets parameter values"""
        return {'order': self._order,
                'trend': self._arima_trend
                }

    def ts_fit(self, suppress=False):
        """Fit ARIMA to the time series data.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        self._prepare_fit()
        self.ts_split()
        ARIMAForecaster._init_trend(self)

        ts_df = self._train_dt.copy()

        # Fit
        self._arima_logger.info("Trying to fit the ARIMA model....")
        # tic
        start = time()
        try:
            if not suppress:
                self._arima_logger.info("...via using parameters\n")
                print_attributes(self)

            self._model = ARIMA(ts_df['y'], order=self._order, freq=self.freq)
            self.model_fit = self._model.fit(trend=self._arima_trend, method='mle', disp=1)
        except (Exception, ValueError):
            self._arima_logger.exception("Exception occurred in the fit...")
            self._arima_logger.error("Please try other parameters!")
            self.model_fit = None

        else:
            # toc
            self._arima_logger.info("Time elapsed: {} sec.".format(time() - start))
            self._arima_logger.info("Model successfully fitted to the data!")
            if not suppress:
                self._arima_logger.info("The model summary: " + str(self.model_fit.summary()))

            # Fitted values
            self._arima_logger.info("Computing fitted values and residuals...")
            self._ar_coef, self._ma_coef = self.model_fit.arparams, self.model_fit.maparams

            self.fittedvalues = self.model_fit.fittedvalues
            # prologue
            if len(self.fittedvalues) != len(self._train_dt):
                self.fittedvalues = pd.DataFrame(
                    index=pd.date_range(ts_df.index[0], ts_df.index[len(ts_df) - 1],
                                        freq=self.freq),
                    columns=['dummy']).join(pd.DataFrame(self.fittedvalues)).drop(['dummy'], axis=1)
                self.fittedvalues = self.fittedvalues.reset_index()
                self.fittedvalues.columns = self._ts_df_cols
                self.fittedvalues.set_index('ds', inplace=True)
                self.fittedvalues.y = self.fittedvalues.y.fillna(method='bfill')

            # Residuals
            super(ARIMAForecaster, self)._residuals()
            self._arima_logger.info("Done.")
            return self

    def ts_diagnose(self):
        """Diagnoses the model.

        In case of ARIMA residual plots are generated.
        Additionally, the kde plot of residuals is returned
        """
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._arima_logger.exception("Model has to be fitted first! Please call ts_fit(...)")
            sys.exit("STOP")

        self.residuals.plot(kind='kde', title='Density')
        print("Residuals statistics")
        print(self.residuals.describe())
        self.plot_residuals()

    def plot_residuals(self):
        """Plot the residuals"""
        fig, axis = super(ARIMAForecaster, self)._plot_residuals(y=np.asarray(self._train_dt['y']),
                                                                 yhat=np.asarray(self.fittedvalues).flatten(),
                                                                 _id="ARIMA")

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        if super(ARIMAForecaster, self)._check_ts_test() < 0:
            return

        n_forecast = len(self._test_dt)

        self._arima_logger.info("Evaluating the fitted ARIMA model on the test data...")
        future = self.model_fit.predict(start=len(self._train_dt.index),
                                        end=len(self._train_dt.index) + n_forecast - 1, dynamic=True)

        self.forecast = pd.Series(future, index=self._test_dt.index)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt['y']) - np.asarray(self.forecast),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._arima_logger.info("RMSE on the test data: {}".format(self.rmse))

        # plot
        if show_plot:
            self.plot_forecast()

    def ts_forecast(self, n_forecast, suppress=False):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(ARIMAForecaster, self)._check_ts_forecast(n_forecast)
        #
        self._arima_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._arima_logger.info("Forecasting next " + str(n_forecast) + str(self.freq))
        #
        future = self.model_fit.predict(start=len(self._train_dt.index),
                                        end=len(self._train_dt.index) + (n_forecast-1), dynamic=True)
        idx_future = self._gen_idx_future(n_forecast=n_forecast)
        self.forecast = pd.Series(future, index=idx_future)

        self.residuals_forecast = None
        # self.plot_forecast()
        return self

    def plot_forecast(self):
        """Plot forecasted values"""
        fig, axis = super(ARIMAForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                yhat=np.asarray(self.fittedvalues).flatten(),
                                                                forecast=self.forecast, _id='ARIMA')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()
