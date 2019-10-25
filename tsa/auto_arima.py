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
import pmdarima as pm
from time import time


class AutoARIMAForecaster(UVariateTimeSeriesClass):
    """Univariate time series child class using pmdarima.auto_arima for forecasting

    Attributes
    ----------
    ref. to https://pypi.org/project/pmdarima/
    https://www.alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html#pmdarima.arima.AutoARIMA
    _start_p: int
        The starting value for p
    _start_q: int
        The starting value for q
    _test: str
       Test for determining the value of d
    _max_p: int
        The maximal value for p: all values between _start_p and this one will be tried out
    _max_q: int
        The maximal value for q: all values between _start_q and this one will be tried out
     _d: int
        The maximum value of d, or the maximum number of non-seasonal differences. If None, this value will be determined.
    _seasonal: bool
        Seasonal component yes/no
    _D:
        The order of the seasonal differencing. If None, the value will automatically be selected based on
        the results of the seasonal_test.
    _start_P: int
         The starting value for P
    _start_Q: int
         The starting value for Q
    _max_P: int
         The maximum value for P
    _max_Q: int
         The maximum value for Q
    _seasonal_periods (m in original package): int
        The period for seasonal differencing, m refers to the number of periods in each season.
        For example, m is 4 for quarterly data, 12 for monthly data, or 1 for annual (non-seasonal) data.
        Default is 1. Note that if m == 1 (i.e., is non-seasonal),
        seasonal will be set to False.
     _aarima_trend: str or iterable, default=’c’, ref. http://www.alkaline-ml.com/pmdarima/1.0.0/modules/generated/pmdarima.arima.auto_arima.html
        Parameter controlling the deterministic trend polynomial A(t). Can be specified as a string where ‘c’
        indicates a constant (i.e. a degree zero component of the trend polynomial),
        ‘t’ indicates a linear trend with time, and ‘ct’ is both.
        Can also be specified as an iterable defining the polynomial as
        in numpy.poly1d, where [1,1,0,1] would denote a+bt+ct3.
    _random : bool, optional (default=False)
        Auto_arima provides the capability to perform a “random search” over a hyper-parameter space.
        If random is True, rather than perform an exhaustive search or stepwise search, only n_fits
        ARIMA models will be fit (stepwise must be False for this option to do anything).
    _n_fits : int, optional (default=10)
        If random is True and a “random search” is going to be performed, n_iter is the number of ARIMA models to be fit.
    _stepwise : bool, optional (default=True)
        Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the
        optimal model parameters.
        The stepwise algorithm can be significantly faster than fitting all (or a random subset of)
        hyper-parameter combinations and is less likely to over-fit the model.
    _information_criterion : str, optional (default=’aic’)
        The information criterion used to select the best ARIMA model.
        One of pmdarima.arima.auto_arima.VALID_CRITERIA, (‘aic’, ‘bic’, ‘hqic’, ‘oob’).
    _scoring : str, optional (default=’mse’)
        If performing validation (i.e., if out_of_sample_size > 0), the metric to use for scoring the
        out-of-sample data. One of {‘mse’, ‘mae’}
    _out_of_sample_size : int, optional (default=0)
        The ARIMA class can fit only a portion of the data if specified, in order to retain an “out of bag” sample score.
        This is the number of examples from the tail of the time series to hold out and use as validation examples.
        The model will not be fit on these samples, but the observations will be added into the model’s endog and exog
        arrays so that future forecast values originate from the end of the endogenous vector.
    _aarima_logger: Logger
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
                 start_p=1,
                 start_q=1,
                 max_p=3,
                 max_q=3,
                 d=None,
                 D=None,
                 start_P=1,
                 start_Q=1,
                 max_P=3,
                 max_Q=3,
                 random = False,
                 n_fits=10,
                 stepwise=True,
                 information_criterion='aic',
                 scoring='mse',
                 out_of_sample_size=0,
                 **kwds):
        """Initializes the object AutoARIMAForecaster"""
        self._aarima_logger = Logger("AutoARIMA")
        self._aarima_seasonal = False
        self._aarima_trend = 'c'
        self._start_p = start_p
        self._start_q = start_q
        self._max_p = max_p
        self._max_q = max_q
        self._d = d
        self._D = D
        self._start_P = start_P
        self._start_Q = start_Q
        self._max_P = max_P
        self._max_Q = max_Q
        self._random = random
        self._n_fits = n_fits
        self._stepwise = stepwise
        self._information_criterion = information_criterion
        self._scoring = scoring
        self._out_of_sample_size = out_of_sample_size
        
        try:
            super(AutoARIMAForecaster, self).__init__(**kwds)
        except TypeError:
            self._aarima_logger.exception("Arguments missing...")

        AutoARIMAForecaster._init_trend(self)
        AutoARIMAForecaster._init_seasonal(self)

        AutoARIMAForecaster.assertions(self)
        
        self._id = 'Auto_ARIMA'

    def _init_trend(self):
        if self._trend is None or self._trend == 'constant':
            self._aarima_trend = 'c'
        elif self._trend == 'linear':
            self._aarima_trend = 't'
        elif self._trend == 'constant linear':
            self._aarima_trend = 'ct'
        elif self._trend in ['additive', 'add']:
            # self._aarima_logger.warning("The trend " + str(self._trend) + " not supported by AutoARIMA! "
            #                                                              "Assuming first order trend")
            self._aarima_trend = 'a+bt'
        elif self._trend in ['multiplicative', 'mul']:
            # self._aarima_logger.warning("The trend " + str(self._trend) + " not supported by AutoARIMA! "
            #                                                              "Assuming first order trend")
            self._aarima_trend = 'a+bt'
    
    def _init_seasonal(self):
        if self._seasonal is None:
            self._aarima_seasonal = False
        if isinstance(self._seasonal, bool):
            self._aarima_seasonal = self._seasonal
        else:
            self._aarima_seasonal = False
            
    def __copy__(self):
        """Copies the object"""
        result = super(AutoARIMAForecaster, self).__copy__()

        result._start_p = self._start_p
        result.start_q = self._start_q
        result._test = self._test
        result._max_p = self._max_p
        result._max_q = self._max_q
        result._d = self._d
        result._aarima_trend = self._aarima_trend
        result._aarima_seasonal = self._aarima_seasonal
        result._D = self._D
        result._start_P = self._start_P
        result._start_Q = self._start_Q
        result._max_P = self._max_P
        result._max_Q = self._max_Q
        result._random = self._random
        result._n_fits = self._n_fits
        result._stepwise = self._stepwise
        result._information_criterion = self._information_criterion
        result._scoring = self._scoring
        result._out_of_sample_size = self._out_of_sample_size

        result._aarima_logger = self._aarima_logger
        return result

    def assertions(self):
        try:
            assert self.hyper_params is None
        except AssertionError:
            self._aarima_logger.exception("Hyper parameters does not make sence for Auto ARIMA! "
                                          "Please specify parameters")
            sys.exit("STOP")

        try:
            assert self._aarima_trend is not None
        except AssertionError:
            self._aarima_logger.exception("Assertion Error, trend cannot be None!")
            sys.exit("STOP")
        try:
            assert isinstance(self._aarima_seasonal, bool)
        except AssertionError:
            self._aarima_logger.exception("Assertion Error, seasonal must be boolean True/False")
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
            elif k == 'start_p':
                self._start_p = v
            elif k == 'max_p':
                self._max_p = v
            elif k == 'start_q':
                self._start_q = v
            elif k == 'max_q':
                self._max_q = v
            elif k == 'd':
                self._d = v
            elif k == 'trend':
                self._aarima_trend = v
            elif k == 'seasonal':
                self._aarima_seasonal = v
            elif k == 'seasonal_periods':
                self._seasonal_periods = v
            elif k == 'start_P':
                self._start_P = v
            elif k == 'max_P':
                self._max_P = v
            elif k == 'start_Q':
                self._start_Q = v
            elif k == 'max_Q':
                self._max_Q = v
            elif k == 'D':
                self._D = v
            elif k == 'random':
                self._random = v
            elif k == 'n_fits':
                self._n_fits = v
            elif k == 'stepwise':
                self._stepwise = v
            elif k == 'information_criterion':
                self._information_criterion = v
            elif k == 'scoring':
                self._scoring = v
            elif k == 'out_of_sample_size':
                self._out_of_sample_size = v
        self.assertions()

        return self

    def get_params_dict(self):
        """Gets parameter values as dictionary"""
        return {'start_p': self._start_p,
                'start_q': self._start_q,
                'test': self._test,
                'max_p': self._max_p,
                'max_q': self._max_q,
                'd': self._d,
                'trend': self._aarima_trend,
                'seasonal': self._aarima_seasonal,
                'seasonal_periods': self._seasonal_periods,
                'D': self._D,
                'start_P': self._start_P,
                'start_Q': self._start_Q,
                'max_P': self._max_P,
                'max_Q': self._max_Q,
                'random': self._random,
                'n_fits': self._n_fits,
                'stepwise': self._stepwise,
                'information_criterion': self._information_criterion,
                'scoring': self._scoring,
                'out_of_sample_size': self. _out_of_sample_size
                }

    def ts_fit(self, suppress=False):
        """Fit Auto ARIMA to the time series data.

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

        """
        Fit
        """
        self._aarima_logger.info("Trying to fit the Auto ARIMA model....")
        # tic
        start = time()
        try:
            if not suppress:
                self._aarima_logger.info("...via using parameters\n")
                print_attributes(self)

            self.model_fit = pm.auto_arima(ts_df,
                                           start_p=self._start_p,
                                           start_q=self._start_q,
                                           test=self._test,
                                           max_p=self._max_p,
                                           m=self._seasonal_periods,
                                           d=self._d,
                                           seasonal=self._aarima_seasonal,
                                           D=self._D,
                                           start_P=self._start_P,
                                           max_P=self._max_P,
                                           trend=self._aarima_trend,
                                           trace=True,
                                           error_action='ignore',
                                           suppress_warnings=True,
                                           stepwise=self._stepwise,
                                           random=self._random,
                                           n_fits=self._n_fits,
                                           scoring=self._scoring,
                                           out_of_sample_size=self._out_of_sample_size,
                                           information_criterion=self._information_criterion)
        except (Exception, ValueError):
            self._aarima_logger.exception("Exception occurred in the fit...")
            self._aarima_logger.warning("Will try to reset some parameters...")
            try:
                self.model_fit = pm.auto_arima(ts_df,
                                               start_p=self._start_p,
                                               start_q=self._start_q,
                                               test=self._test,
                                               max_p=self._max_p,
                                               m=1,
                                               d=0,
                                               seasonal=self._aarima_seasonal,
                                               D=0,
                                               start_P=self._start_P,
                                               max_P=self._max_P,
                                               trend=self._aarima_trend,
                                               trace=True,
                                               error_action='ignore',
                                               suppress_warnings=True,
                                               stepwise=self._stepwise,
                                               random=self._random,
                                               n_fits=self._n_fits,
                                               scoring=self._scoring,
                                               out_of_sample_size=self._out_of_sample_size,
                                               information_criterion=self._information_criterion)
            except (Exception, ValueError):
                self._aarima_logger.exception("Exception occurred")
                self._aarima_logger.error("Please try other parameters!")
                self.model_fit = None

        else:
            # toc
            self._aarima_logger.info("Time elapsed: {} sec.".format(time() - start))
            #
            self._aarima_logger.info("Model successfully fitted to the data!")
            self._aarima_logger.info("The chosen model AIC: " + str(self.model_fit.aic()))

            # Fitted values
            self._aarima_logger.info("Computing fitted values and residuals...")
            self.fittedvalues = pd.Series(self.model_fit.predict_in_sample(start=0, end=(len(ts_df) - 1)),
                                          index=ts_df.index)
            # Residuals
            super(AutoARIMAForecaster, self)._residuals()
            
            self._aarima_logger.info("Done.")
            return self

    def ts_diagnose(self):
        """Diagnose the model"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._aarima_logger.exception("Model has to be fitted first! Please call ts_fit(...)")
            sys.exit("STOP")

        self.model_fit.plot_diagnostics(figsize=(9, 3.5))
        self.plot_residuals()

    def plot_residuals(self):
        """Plot the residuals"""
        fig, axis = super(AutoARIMAForecaster, self)._plot_residuals(y=np.asarray(self._train_dt['y']),
                                                                     yhat=np.asarray(self.fittedvalues),
                                                                     _id=" Auto ARIMA")

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        if super(AutoARIMAForecaster, self)._check_ts_test() < 0:
            return

        n_forecast = len(self._test_dt)

        self._aarima_logger.info("Evaluating the fitted ARIMA model on the test data...")
        future, confint = self.model_fit.predict(n_periods=n_forecast, return_conf_int=True)
        self.forecast = pd.Series(future, index=self._test_dt.index)
        self.lower_conf_int = pd.Series(confint[:, 0], index=self._test_dt.index)
        self.upper_conf_int = pd.Series(confint[:, 1], index=self._test_dt.index)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt['y']) - np.asarray(self.forecast),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._aarima_logger.info("RMSE on test data: {}".format(self.rmse))

        # plot
        if show_plot:
            self.plot_forecast()

    def ts_forecast(self, n_forecast, suppress=False):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(AutoARIMAForecaster, self)._check_ts_forecast(n_forecast)
        #
        self._aarima_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._aarima_logger.info("Forecasting next " + str(n_forecast) + str(self.ts_df.index.freq))
        #
        future, confint = self.model_fit.predict(n_periods=n_forecast, return_conf_int=True)
        idx_future = self._gen_idx_future(n_forecast=n_forecast)
        self.forecast = pd.Series(future, index=idx_future)
        if self.lower_conf_int is None and self.upper_conf_int is None:
            self.lower_conf_int = pd.Series(confint[:, 0], index=idx_future)
            self.upper_conf_int = pd.Series(confint[:, 1], index=idx_future)
        else:
            self.lower_conf_int = pd.concat([self.lower_conf_int, pd.Series(confint[:, 0], index=idx_future)], axis=0)
            self.upper_conf_int = pd.concat([self.upper_conf_int, pd.Series(confint[:, 1], index=idx_future)], axis=0)

        self.residuals_forecast = None
        # self.plot_forecast()
        return self

    def plot_forecast(self):
        """Plot forecasted values"""
        fig, axis = super(AutoARIMAForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                    yhat=np.asarray(self.fittedvalues),
                                                                    forecast=self.forecast, _id='Auto ARIMA')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()
