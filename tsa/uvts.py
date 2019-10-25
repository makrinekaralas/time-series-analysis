__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

from tsa import Logger

import sys
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from stldecompose import decompose
from scipy import stats
from abc import abstractmethod
import tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs
from itertools import compress
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


class UVariateTimeSeriesClass(object):
    """Univariate time series class

    Attributes
    ----------
    _ts_df_cols: list
        internal column names for dataframe that will be input to model
    ts_df: dataframe
        time series data frame
    freq: int
       frequency of time series; python format
    fill_method: str
       filling method for resampled data. Possible are 'ffill' and 'interp1d'
    n_test: int
         number of units (defined by frequency, e.g. 6 days) to use as test data. 0 would mean no test data is generated.
    n_val: int
        similar to n_test, for validation
    time_format: str
        time format if time series data needs to be brought into datetime

    _mode: str
        defines the mode as 'test' or 'forecast'
    _train_dt: dataframe
        training data
    _test_dt: dataframe
        test data
    _val_dt: dataframe
        validation data

    model_fit:
      fitted model
    fittedvalues: series
       computed fitted values
    residuals: series
       residuals
    rmse: float
       RMSE on test set (test data and the forecast on test data)
    _gs: GridSearchClass
       The grid search class for model optimization in case hyper_parameters are specified
    hyper_params: dictionary
       The dictionary of hyper parameters or None if no model optimization wished
    best_model: dictionary
        The best model resulted from the grid search

    upper_whisker_res: float
       upper whisker for residuals
    lower_conf_int: series
      lower confidence interval
    upper_conf_int: series
      upper confidence interval

    _trend: str or iterable, default=’c’
        in AutoARIMA:
            ref. http://www.alkaline-ml.com/pmdarima/1.0.0/modules/generated/pmdarima.arima.auto_arima.html
            Parameter controlling the deterministic trend polynomial A(t). Can be specified as a string where ‘c’
            indicates a constant (i.e. a degree zero component of the trend polynomial),
            ‘t’ indicates a linear trend with time, and ‘ct’ is both.
            Can also be specified as an iterable defining the polynomial as
            in numpy.poly1d, where [1,1,0,1] would denote a+bt+ct3.
        in ARIMA:
            A parameter for controlling a model of the deterministic trend as one of ‘nc’ or ’c’.
            ‘c’ includes constant trend, ‘nc’ no constant for trend.
        in SARIMA:
            A parameter for controlling a model of the deterministic trend as one of ‘n’,’c’,’t’,’ct’ for no trend,
            constant, linear, and constant with linear trend, respectively.
        in ExponentialSmoothing:
            The type of trend component, as either “add” for additive or “mul” for multiplicative.
            Modeling the trend can be disabled by setting it to None

    _test: list or str
        in ARIMA:
           list of possible tests for determining d
        in AutoARIMA.
           test for determining the value of d, e.g. 'adf'

    _seasonal: bool or str
        in AutoARIMA
            Seasonal component yes/no

        in ExponentialSmoothing:
            The type of seasonal component, as either “add” for additive or “mul” for multiplicative.
            Modeling the seasonal component can be disabled by setting it to None

    _seasonal_periods: int
         The number of time steps in a seasonal period, e.g. 12 for 12 months in a yearly seasonal structure

    forecast: series
      computed forcatsed values
    residuals_forecast: series
      residuals between forecasted and real values. Note, this variable exist only if test data existed

    Methods
    -------
    ts_transform()
         Transforms time series using log10 or box-cox
    ts_resample()
         Resamples time series at the chosen frequency freq
    ts_test()
         Evaluates fitted model on the test data, if this one has been generated
    ts_forecast()
         Forecasts time series and plots the results
    ts_decompose()
         Decomposes time series in _arr_seasonal, _arr_trend, residual(irregular) and _arr_baseline,
         and plots the results
    plot_decompose()
         Plots the results of ts_decompose()
    difference()
        Differences the time series given the lag (parameter interval)
    rolling_mean()
        Computes moving average given the window size
    rolling_variance()
        Computes moving variance given the window size
    test_adf():
         ADF test for stationarity
    test_kpss():
         KPSS test for stationarity
    ndiff()
        Determines value for diff parameter d
        All tests given in the parameter tests are applied
    acf_plots()
        Generates autocorrelation plots
    pacf_plots()
        Generates partial correlation plots

    Helper methods:
    -------
    _plot_residuals()
         Residual plots helper function
    _plot_forecast()
         Helper function for plotting forecasted time-series
    _prepare_fit()
         Prepares ts_fit of child class. Supposed to be called by a child class
    _residuals()
         Helper function for calculating residuals. Supposed to be called by a child class
    _check_ts_test()
         Checks for test. Supposed to be called by a child class
    _check_ts_forecast()
         Checks for forecast. Supposed to be called by a child class
    """

    def __init__(self, ts_df, time_format="%Y-%m-%d %H:%M:%S", freq='D',
                 fill_method='ffill',
                 n_test=0, n_val=0,
                 hyper_params=None,
                 test='adf',
                 trend=None,
                 seasonal=False,
                 seasonal_periods=1,
                 **kwds):
        """Initializes the object UVariateTimeSeriesForecaster"""
        self._ts_df_cols = ['ds', 'y']

        self.ts_df = ts_df
        self.time_format = time_format
        self.freq = freq
        self.fill_method = fill_method.lower()
        self.n_test = int(n_test)
        self.n_val = int(n_val)
        self.transform = None
        self._boxcox_lmbda = None

        self._mode = ''

        self._train_dt = None
        self._test_dt = None
        self._val_dt = None

        self.model_fit = None
        self.fittedvalues = None
        self.residuals = None
        self.rmse = 0
        self._gs = tsa.GridSearchClass()
        self.hyper_params = hyper_params
        self.best_model = dict()

        """
        self.rmse_test = 0
        self.rmse_val = 0
        """

        self.upper_whisker_res = None
        self.lower_conf_int = None
        self.upper_conf_int = None

        self.forecast = None
        self.residuals_forecast = None

        self._res_decomp = None
        self._arr_seasonal = None
        self._arr_trend = None
        self._arr_baseline = None

        self._test = test
        self._trend = trend
        if self._trend is not None:
            self._trend = self._trend.lower()
        self._seasonal = seasonal
        if isinstance(self._seasonal, str):
            self._seasonal = self._seasonal.lower()
        self._seasonal_periods = seasonal_periods

        self._uvts_cls_logger = Logger('uvts_cls')

        UVariateTimeSeriesClass.assertions(self)
        # work with ts_df
        self.ts_df = self.ts_df.reset_index()
        self.ts_df.columns = self._ts_df_cols
        self.ts_df['y'] = self.ts_df['y'].apply(np.float64, errors='coerce')
        self.ts_df.set_index('ds', inplace=True)
        self._uvts_cls_logger.info(
            "Received time series data of range: " + str(min(self.ts_df.index)) + ' - ' + str(
                max(self.ts_df.index)) + " and shape: " + str(self.ts_df.shape))

        if not isinstance(self.ts_df.index, pd.DatetimeIndex):
            self._uvts_cls_logger.warning("Time conversion required...")
            self.ts_df = self.ts_df.reset_index()
            try:
                self.ts_df['ds'] = self.ts_df['ds'].apply(
                    lambda x: datetime.datetime.strptime(
                        str(x).translate({ord('T'): ' ', ord('Z'): None})[:-1],
                        self.time_format))
            except ValueError as e:
                self._uvts_cls_logger.warning("Zulu time conversion not successful: {}".format(e))
                self._uvts_cls_logger.warning("Will try without assuming zulu time...")
                try:
                    self.ts_df['ds'] = self.ts_df['ds'].apply(
                        lambda x: datetime.datetime.strptime(str(x), self.time_format))
                except ValueError as e:
                    self._uvts_cls_logger.info("Time conversion not successful. Check your time_format: {}".format(e))
                    sys.exit("STOP")
                else:
                    self._uvts_cls_logger.info("Time conversion successful!")
            else:
                self._uvts_cls_logger.info("Time conversion successful!")
            # set index
            self.ts_df.set_index('ds', inplace=True)
        #
        self.ts_df.index = pd.to_datetime(self.ts_df.index)
        self.ts_df.sort_index(inplace=True)
        # resample
        self.ts_resample()
        UVariateTimeSeriesClass.assertions(self, post=True)
        #
        if self.n_val > len(self.ts_df) - self.n_test:
            self.n_val = len(self.ts_df) - self.n_test

        if self.n_test == 0 and self.n_val == 0:
            self._mode = 'forecast'
        elif self.n_test > 0:
            self._mode = 'test'
        elif self.n_test == 0 and self.n_val > 0:
            self._mode = 'validate'
        
        # delegate just for good programming style here
        super(UVariateTimeSeriesClass, self).__init__(**kwds)

    def assertions(self, post=False):
        
        if post:
            try:
                assert 0 <= self.n_test < len(self.ts_df)
            except AssertionError:
                self._uvts_cls_logger.exception("Assertion exception, invalid value for n_test!")
                sys.exit("STOP")
            #  
            try:
                assert 0 <= self.n_val < len(self.ts_df)
            except AssertionError:
                self._uvts_cls_logger.exception("Assertion exception, invalid value for n_val!")
                sys.exit("STOP")
        else:
            try:
                assert self.fill_method in ['ffill', 'interp1d']
            except AssertionError:
                self._uvts_cls_logger.exception("Assertion exception, fill method not recognized! "
                                                "'ffill' will be used. ")
            else:
                self.fill_method = 'ffill'

            try:
                assert pd.DataFrame(self.ts_df).shape[1] <= 2
            except AssertionError:
                self._uvts_cls_logger.exception(
                    "Time series must be uni-variate. "
                    "Hence, at most a time columns and a column of numeric values are expected!")
                sys.exit("STOP")

            try:
                self._trend is None or (isinstance(self._trend, str) and self._trend in ['constant', 'linear',
                                                                                         'constant linear', 'additive',
                                                                                         'add',
                                                                                         'multiplicative', 'mul'])
            except AssertionError:
                self._uvts_cls_logger.exception("Assertion exception occurred, invalid value for trend! "
                                                "Choose between None or  "
                                                "['constant', 'linear ','constant linear', "
                                                "'additive', 'add , 'multiplicative', 'mul'] ")
                sys.exit("STOP")

            try:
                self._seasonal is None or isinstance(self._seasonal, bool) or (isinstance(self._seasonal, str) and
                                                                               self._seasonal in ['additive', 'add',
                                                                                                  'multiplicative', 'mul'])
            except AssertionError:
                self._uvts_cls_logger.exception("Assertion exception occurred, invalid value for seasonal! "
                                                "Choose between True/False, None or  "
                                                "['additive', 'add , 'multiplicative', 'mul'] ")
                sys.exit("STOP")

    def __copy__(self):
        """Copies the object"""

        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def ts_transform(self, transform):
        """Transforms time series via applying casted 'transform'. Right now 'log10' and 'box-cox' possible."""
        try:
            assert transform.lower().strip() in ['log10', 'box-cox']
        except AssertionError:
            self._uvts_cls_logger.error(
                "transform should be in ['log10', 'box-cox'] or empty. Assuming no transform! "
                "Hence, if you get bad results, you would like maybe to choose e.g., log10 here.")
            self._uvts_cls_logger.exception("Assertion exception occurred, transform")
            self.transform = None
        else:
            self.transform = transform.lower()
            # transform
            if sum(self.ts_df['y'] > 0) < len(self.ts_df['y']):
                self._uvts_cls_logger.warning("Zero, negative, or both values present in your data. Transformation will not be used!")
                return self
            if self.transform == 'log10':
                try:
                    self.ts_df['y'] = self.ts_df['y'].apply(np.log10)
                except ValueError:
                    self._uvts_cls_logger.exception("log10 transformation did not work! Possibly negative "
                                                    "values present?")
            elif self.transform == 'box-cox':
                if input("Do you want to provide lambda for box.cox? y/n?").strip().lower() == 'y':
                    self._boxcox_lmbda = float(input())
                else:
                    self._boxcox_lmbda = None
                try:
                    if self._boxcox_lmbda is None:
                        bc, lmbda_1 = stats.boxcox(self.ts_df['y'], lmbda=self._boxcox_lmbda)
                        self.ts_df['y'] = stats.boxcox(self.ts_df['y'], lmbda=lmbda_1)
                    else:
                        self.ts_df['y'] = stats.boxcox(self.ts_df['y'], lmbda=self._boxcox_lmbda)
                except ValueError:
                    self._uvts_cls_logger.exception("box-cox transformation did not work! "
                                                    "Possibly negative values present or bad lambda?")
        return self

    def set_frequency(self, new_freq):
        """Sets new frequency and resamples time series to that new frequency"""
        self.freq = new_freq
        self.ts_resample()

    def ts_check_frequency(self):
        """Checks the frequency of time series"""
        if self.ts_df.index.freq is None:
            self._uvts_cls_logger.info("No specific frequency detected.")
            self._uvts_cls_logger.info("Frequency chosen in initialization: " + str(
                self.freq) + " enter 'n' and call ts_resample() if you are satisfied with this value.")
            if input("Should a histogram of time deltas be plotted y/n?").strip().lower() == 'y':
                ff = pd.Series(self.ts_df.index[1:(len(self.ts_df))] - self.ts_df.index[0:(len(self.ts_df) - 1)])
                ff = ff.apply(lambda x: int(x.total_seconds() / (60 * 60)))
                plt.hist(ff, bins=120)
                plt.xlabel("Rounded time delta [H]")
                plt.ylabel("Frequency of occurrence")
                self._uvts_cls_logger.info(ff.value_counts())
                self._uvts_cls_logger.info("Should hourly frequency not fit, choose a reasonable frequency and call "
                                           "set_frequency(new_freq)")
            else:
                pass
        else:
            self._uvts_cls_logger.info("Time series frequency: " + str(self.ts_df.index.freq))

    def ts_resample(self):
        """Brings original time series to the chosen frequency freq"""
        try:
            ts_freq = pd.DataFrame(
                index=pd.date_range(self.ts_df.index[0], self.ts_df.index[len(self.ts_df) - 1], freq=self.freq),
                columns=['dummy'])
        except ValueError:
            self._uvts_cls_logger.exception("Exception occurred, possibly incompatible frequency!")
            sys.exit("STOP")

        if self.fill_method == 'ffill':
            self.ts_df = ts_freq.join(self.ts_df).drop(['dummy'], axis=1)
            self.ts_df.y = self.ts_df.y.fillna(method='ffill')
            # if np.isnan ( self.ts_df.y ).any ():
            #    self.ts_df.y = self.ts_df.y.fillna ( method='bfill' )
        else:  # interp
            xp = np.linspace(0, self.ts_df.size, self.ts_df.size, endpoint=False)
            fp = self.ts_df['y']
            # join
            self.ts_df = ts_freq.join(self.ts_df).drop(['dummy'], axis=1)
            # pick new points
            x = np.linspace(0, ts_freq.size, ts_freq.size, endpoint=False)
            x = x[self.ts_df['y'].isna()]
            print(x.size)
            print(x)

            # put the values
            self.ts_df.y[self.ts_df['y'].isna()] = np.interp(x, xp, fp)

        if np.isnan(self.ts_df.y).any():
            self._uvts_cls_logger.warning("Some NaN found, something went wrong, check the data!")
            sys.exit("STOP")

        self._uvts_cls_logger.info("Time series resampled at frequency: " + str(self.ts_df.index.freq) +
                                   ". New shape of the data: " + str(self.ts_df.shape))
        self._uvts_cls_logger.info("Using time series data of range: " + str(min(self.ts_df.index)) + ' - ' + str(
            max(self.ts_df.index)) + " and shape: " + str(self.ts_df.shape))

        return self

    def ts_split(self):
        """Prepares data for different modes: train, test, validate, test and validate, forecast"""

        if self.ts_df.index.freq is None:
            self._uvts_cls_logger.warning("Time series exhibit no frequency. Calling ts_resample()...")
            try:
                self.ts_resample()
            except ValueError:
                self._uvts_cls_logger.error("Resample did not work! Error:" + str(sys.exc_info()[0]))

        ts_df = self.ts_df

        if self._mode == 'forecast':
            self._train_dt = ts_df
            self._test_dt, self._val_dt = None, None
        elif self._mode == 'test and validate':
            if self._test_dt is not None:
                self._train_dt = pd.concat([self._train_dt, self._test_dt], axis=0)
                self._test_dt = self._val_dt
                self._val_dt = None
            else:
                self._uvts_cls_logger.error("Something is wrong: mode!")
        else:
            # split
            ts_test_df = pd.DataFrame()
            ts_val_df = pd.DataFrame()
            #
            ts_df = ts_df.reset_index()
            ts_df.columns = self._ts_df_cols

            if self._mode == 'test' and self.n_val == 0:
                ts_test_df = ts_df.copy()
                #
                ts_df = pd.DataFrame(ts_df.loc[:(len(ts_df) - 1 - self.n_test), ])
                ts_df.set_index('ds', inplace=True)
                # test
                ts_test_df = pd.DataFrame(ts_test_df.loc[(len(ts_test_df) - self.n_test):, ])
                ts_test_df.set_index('ds', inplace=True)
            elif self._mode == 'validate':
                ts_val_df = ts_df.copy()
                #
                ts_df = pd.DataFrame(ts_df.loc[:(len(ts_df) - 1 - self.n_val), ])
                ts_df.set_index('ds', inplace=True)
                # val
                ts_val_df = pd.DataFrame(ts_val_df.loc[(len(ts_val_df) - self.n_val):, ])
                ts_val_df.set_index('ds', inplace=True)
            elif self._mode == 'test' and self.n_val > 0:
                ts_test_df = ts_df.copy()
                ts_val_df = ts_df.copy()
                #
                ts_df = pd.DataFrame(ts_df.loc[:(len(ts_df) - 1 - self.n_test - self.n_val), ])
                ts_df.set_index('ds', inplace=True)
                # test
                ts_test_df = pd.DataFrame(ts_test_df.loc[(len(ts_test_df) -
                                                          self.n_test - self.n_val):(
                                                                 len(ts_test_df) - self.n_val - 1)])
                ts_test_df.set_index('ds', inplace=True)
                # val
                ts_val_df = pd.DataFrame(ts_val_df.loc[(len(ts_val_df) - self.n_val):, ])
                ts_val_df.set_index('ds', inplace=True)

            # now set
            self._train_dt = ts_df
            if not ts_test_df.empty:
                self._test_dt = ts_test_df
            if not ts_val_df.empty:
                self._val_dt = ts_val_df

        return self

    @staticmethod
    def compute_ci(yhat, yhat_var, ci_level):
        """Easy compute of confidence intervals"""
        z_mapping = {0.95: 1.96,
                     0.99: 2.58}
        z = z_mapping[ci_level]

        ci_lower = yhat - yhat_var * z
        ci_upper = yhat + yhat_var * z

        return ci_lower, ci_upper

    def _prepare_fit(self):
        """Helper function ro prepare ts_fit"""
        self.lower_conf_int, self.upper_conf_int, self.upper_whisker_res = None, None, None
        self.model_fit = None
        self.residuals, self.residuals_forecast, self.fittedvalues = None, None, None

    def _residuals(self):
        """Helper function to calculate residuals"""
        if self.model_fit is None:
            self._uvts_cls_logger.error("No model has been fitted, residuals cannot be computed!")
            sys.exit("STOP")

        try:
            # use fittedvalues to fill in the model dictionary
            self.residuals = pd.Series(np.asarray(self._train_dt['y']) - np.asarray(self.fittedvalues).flatten(),
                                       index=self._train_dt['y'].index)
            self.upper_whisker_res = self.residuals.mean() + 1.5 * (
                    self.residuals.quantile(0.75) - self.residuals.quantile(0.25))
        except (KeyError, AttributeError):
            self._uvts_cls_logger.exception("Exception occurred: Model was not fitted or ts has other structure")

        return self

    def _plot_residuals(self, y, yhat, _id):
        """Helper function to plot the residuals"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._uvts_cls_logger.exception("Model has to be fitted first! Please call ts_fit(...)")

        fig, axes = plt.subplots(2, 1, figsize=(20, 5), sharex=True)

        axes[0].plot(pd.Series(yhat, index=self._train_dt.index), color='y', linewidth=2.0)
        axes[0].plot(pd.Series(y, index=self._train_dt.index), color='b')

        axes[0].set_ylabel("Model Fit")
        axes[0].set_title("Real (blue) and estimated values, " + str(_id))
        #
        axes[1].plot(self.residuals, color="r")
        """
        if self.forecast is not None and self.residuals_forecast is None \
                and self.lower_conf_int is not None and self.upper_conf_int is not None:
            axes[0].fill_between(self.lower_conf_int.index, self.lower_conf_int, self.upper_conf_int, color='k',
                                 alpha=.15)
        """
        if self.lower_conf_int is not None and self.upper_conf_int is not None:
            axes[0].fill_between(self.lower_conf_int.index, self.lower_conf_int, self.upper_conf_int, color='k',
                                 alpha=.15)
        if self.upper_whisker_res is not None:
            axes[1].axhline(y=self.upper_whisker_res, xmin=0, xmax=1, color='m', label='upper_whisker', linestyle='--',
                            linewidth=1.5)
            axes[1].axhline(y=-self.upper_whisker_res, xmin=0, xmax=1, color='m', label='upper_whisker', linestyle='--',
                            linewidth=1.5)

        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Difference between model output and the real data and +/- upper whisker, ' + str(_id))

        return fig, axes

    def _check_ts_test(self):
        """Check before ts_test in child class is called"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._uvts_cls_logger.exception("Model has to be fitted first! Please call ts_fit(...)")

        try:
            assert self._test_dt is not None
        except(KeyError, AssertionError):
            self._uvts_cls_logger.exception("Nothing to test. "
                                            "Call ts_forecast() or specify amount of test data "
                                            "when initializing the object.")
            return -1
        else:
            # self._mode = 'test'
            return 0

    def _check_ts_forecast(self, n_forecast):
        """Check before ts_forecast in child class is called"""
        #
        try:
            n_forecast = int(n_forecast)
            assert 0 < n_forecast < len(self._train_dt)
        except AssertionError:
            self._uvts_cls_logger.exception("Number of periods to be forecasted is too low, too high or not numeric!")
        except ValueError:
            self._uvts_cls_logger.exception("n_forecast must be convertible to int type!")

        return n_forecast

    def _gen_idx_future(self, n_forecast):
        """Generate the time axis for future data"""
        idx_future = None
        if self.freq == 'S':
            idx_future = pd.date_range(start=max(self._train_dt.index) + datetime.timedelta(
                seconds=1),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           seconds=n_forecast), freq='S')
        elif self.freq == 'min':
            idx_future = pd.date_range(start=max(self._train_dt.index) + datetime.timedelta(
                minutes=1),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           minutes=n_forecast), freq='min')
        elif self.freq == 'H':
            idx_future = pd.date_range(start=max(self._train_dt.index) + datetime.timedelta(
                hours=1),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           hours=n_forecast), freq='H')
        elif self.freq == 'D':
            idx_future = pd.date_range(start=max(self._train_dt.index) + datetime.timedelta(
                days=1),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           days=n_forecast), freq='D')
        elif self.freq == 'W':
            idx_future = pd.date_range(start=max(self._train_dt.index) + datetime.timedelta(
                weeks=1),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           weeks=n_forecast), freq='W')
        elif self.freq == 'M' or self.freq == 'MS':
            idx_future = pd.date_range(start=max(self._train_dt.index) + relativedelta(months=+1),
                                       end=max(self._train_dt.index) + relativedelta(months=+n_forecast),
                                       freq=self.freq)
        return idx_future

    def _prepare_forecast(self, yhat, forecast):
        # forecast
        forecast = forecast.reset_index()
        forecast.columns = self._ts_df_cols
        forecast.set_index('ds', inplace=True)
        #
        vals = list()
        vals.append(yhat[-1])
        for i in range(len(forecast['y'])):
            vals.append(forecast['y'][i])

        idx = list()
        idx.append(self._train_dt.index[-1])
        for i in range(len(forecast.index)):
            idx.append(forecast.index[i])
        #
        return pd.Series(vals, index=idx)

    def _plot_forecast(self, y, yhat, forecast, _id):
        """Helper function to plot forecasted values"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._uvts_cls_logger.exception("Model has to be fitted first! Please call ts_fit(...)")
            sys.exit("STOP")
        #
        try:
            assert self.forecast is not None
        except AssertionError:
            self._uvts_cls_logger.exception("Neither ts_test(...) nor ts_forecast(...) have been called yet!")
            sys.exit("STOP")

        fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True)
        #
        axes[0].plot(pd.Series(yhat, index=self._train_dt.index), color='y', linewidth=2.0)
        axes[0].plot(pd.Series(y, index=self._train_dt.index), color='b', linewidth=1.0)
        #
        if self.residuals_forecast is not None:
            axes[0].plot(self.ts_df, color='b')
        forecast = self._prepare_forecast(yhat=yhat, forecast=forecast)
        axes[0].plot(forecast, color='orange', linewidth=2.0)
        #
        if self.lower_conf_int is not None and self.upper_conf_int is not None:
            axes[0].fill_between(self.lower_conf_int.index,
                                 self.lower_conf_int,
                                 self.upper_conf_int,
                                 color='k', alpha=.15)
        axes[0].set_ylabel("Fit and Forecast/Validation")
        axes[0].set_title("Real (blue), estimated (yellow) and forecasted values, " + str(_id))
        #
        if self.residuals_forecast is not None:
            axes[1].plot(pd.concat([self.residuals, self.residuals_forecast], axis=0), color='r')
        axes[1].plot(self.residuals, color="r")

        if self.upper_whisker_res is not None:
            axes[1].axhline(y=self.upper_whisker_res,
                            xmin=0,
                            xmax=1, color='m',
                            label='upper_whisker',
                            linestyle='--', linewidth=1.5)
            axes[1].axhline(y=-self.upper_whisker_res,
                            xmin=0,
                            xmax=1, color='m',
                            label='upper_whisker',
                            linestyle='--', linewidth=1.5)
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Difference between model output and the real data both, for fitted "
                          "and forecasted and +/- upper whisker or confidence intervals, " + str(_id))

        return fig, axes

    def ts_decompose(self, params=None):
        """Decomposes time series"""
        self._res_decomp = None
        self._arr_seasonal = None
        self._arr_trend = None
        self._arr_baseline = None
        self.residuals = None

        if params is None:
            params = dict({'model': 'additive',
                           'freq': 1})
        try:
            assert isinstance(params, dict)
        except AssertionError:
            self._uvts_cls_logger.exception("Dictionary is expected for parameters!")
            sys.exit("STOP")

        try:
            assert 'model' in list(params.keys())
        except AssertionError:
            self._uvts_cls_logger.exception("Unexpected dictionary keys. At least decomposition "
                                            "model must be supplied!")
            sys.exit("STOP")

        try:
            assert params['model'].lower() in ['additive', 'multiplicative']
        except AssertionError:
            self._uvts_cls_logger.exception("Unexpected value for the parameter 'model'! "
                                            "Choose from ['additive', 'multiplicative']")
            sys.exit("STOP")
        else:
            params['model'] = params['model'].lower()

        if 'freq' not in list(params.keys()):
            params['freq'] = 1

        try:
            ts2decomp = self.ts_df
            if 'from' in list(params.keys()):
                ts2decomp = ts2decomp[ts2decomp.index >= datetime.datetime.strptime(params['from'], self.time_format)]
            if 'to' in list(params.keys()):
                ts2decomp = ts2decomp[ts2decomp.index <= datetime.datetime.strptime(params['to'], self.time_format)]
            try:
                assert ts2decomp.size > 0
            except AssertionError:
                self._uvts_cls_logger.exception("Empty time series resulted, please check your parameters!")
                sys.exit("STOP")

            if ts2decomp.index.freq is not None:
                res = seasonal_decompose(ts2decomp.loc[:, 'y'], model=params['model'])
            else:
                res = seasonal_decompose(ts2decomp.loc[:, 'y'], model=params['model'], freq=params['freq'])

        except ValueError:
            self._uvts_cls_logger.exception("ValueError, seasonal_decompose error")
        else:
            self._res_decomp = res
            self._arr_seasonal = res.seasonal
            self._arr_trend = res.trend
            self._arr_baseline = self._arr_seasonal + self._arr_trend
            self.residuals = res.resid
            self.upper_whisker_res = self.residuals.mean() + 1.5 * (
                    self.residuals.quantile(0.75) - self.residuals.quantile(0.25))
            self.plot_decompose()

    def ts_stl_decompose(self, params=None):
        self._res_decomp = None
        self._arr_seasonal = None
        self._arr_trend = None
        self._arr_baseline = None
        self.residuals = None

        if params is None:
            params = dict({'period': 12})
        try:
            assert isinstance(params, dict)
        except AssertionError:
            self._uvts_cls_logger.exception("Dictionary is expected for parameters!")
            sys.exit("STOP")

        try:
            assert 'period' in list(params.keys())
        except AssertionError:
            self._uvts_cls_logger.exception("Unexpected dictionary keys. At least decomposition "
                                            "period must be supplied!")
            sys.exit("STOP")

        try:
            assert isinstance(params['period'], int)
        except AssertionError:
            self._uvts_cls_logger.exception("Unexpected value for the parameter 'period'! "
                                            "Integer expected")
            sys.exit("STOP")

        try:
            ts2decomp = self.ts_df
            if 'from' in list(params.keys()):
                ts2decomp = ts2decomp[ts2decomp.index >= datetime.datetime.strptime(params['from'], self.time_format)]
            if 'to' in list(params.keys()):
                ts2decomp = ts2decomp[ts2decomp.index <= datetime.datetime.strptime(params['to'], self.time_format)]
            try:
                assert ts2decomp.size > 0
            except AssertionError:
                self._uvts_cls_logger.exception("Empty time series resulted, please check your parameters!")
                sys.exit("STOP")

            res = decompose(ts2decomp, period=params['period'])

        except ValueError:
            self._uvts_cls_logger.exception("ValueError, stl_decompose error")
        else:
            self._res_decomp = res
            self._arr_seasonal = res.seasonal
            self._arr_trend = res.trend
            self._arr_baseline = self._arr_seasonal + self._arr_trend
            self.residuals = res.resid
            self.upper_whisker_res = np.asarray(self.residuals.mean() + 1.5 * (
                    self.residuals.quantile(0.75) - self.residuals.quantile(0.25)))
            self.plot_decompose()

    def plot_decompose(self):
        """Plots the results of time series decomposition"""
        try:
            assert self._arr_seasonal is not None
        except AssertionError:
            self.ts_decompose()

        fig, axes = plt.subplots(5, 1, figsize=(20, 9), sharex=True)
        axes[0].plot(self._res_decomp.observed)
        axes[0].set_ylabel("Original")
        #
        axes[1].plot(self._arr_trend)
        axes[1].set_ylabel("Trend")
        #
        axes[2].plot(self._arr_seasonal)
        axes[2].set_ylabel("Seasonal")
        #
        axes[3].plot(self._arr_baseline)
        axes[3].set_ylabel("Baseline")
        #
        axes[4].plot(self.residuals)
        axes[4].set_ylabel("Residuals")
        #
        if self.upper_whisker_res is not None:
            axes[4].axhline(y=self.upper_whisker_res,
                            xmin=0,
                            xmax=1, color='m',
                            label='upper_whisker',
                            linestyle='--', linewidth=1.5)
            axes[4].axhline(y=-self.upper_whisker_res,
                            xmin=0,
                            xmax=1, color='m',
                            label='upper_whisker',
                            linestyle='--', linewidth=1.5)

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def difference(self, lag=1):
        diff = list()
        for i in range(lag, len(self.ts_df)):
            value = self.ts_df['y'][i] - self.ts_df['y'][i - lag]
            diff.append(value)
        return pd.Series(diff)

    def rolling_mean(self, window=10):
        return self.ts_df.rolling(window=window).mean()

    def rolling_variance(self, window=10):
        return self.ts_df.rolling(window=window).std()

    def test_adf(self):
        """Performs Dickey-Fuller test for stationarity"""

        dftest = adfuller(self.ts_df['y'], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
        if dftest[0] > dftest[4]['5%']:
            print(
                "Test statistic greater than critical value at 5% --> series seems to be not stationary. "
                "Look at critical values at 1% and 10% too, ideally they also should be less than test statistic.")
        else:
            print(
                "Test statistic less than critical value at 5% --> series seems to be stationary. "
                "Look at critical values at 1% and 10% too, ideally they also should be greater than test statistic.")

    def test_kpss(self):
        """Performs Kwiatkowski-Phillips-Schmidt-Shin test for stationarity"""

        kpsstest = kpss(self.ts_df['y'], regression='c')
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
        for key, value in kpsstest[3].items():
            kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)
        if kpsstest[0] > kpsstest[3]['5%']:
            print(
                "Test statistic greater than critical value at 5% --> series seems to be not stationary. "
                "Look at critical values at 1% and 10% too, ideally they also should be greater than test statistic.")
        else:
            print(
                "Test statistic less than critical value at 5% --> series seems to be stationary. "
                "Look at critical values at 1% and 10% too, ideally they also should be less than test statistic.")

    def ndiff(self, tests=['kpss', 'adf', 'pp'], alpha=0.05, max_d=2):
        """Returns p-values to decide for the value of d-differentiation

        list of tests given in tests parameter are applied.
        """
        try:
            assert sum([i in ['kpss', 'adf', 'pp'] for i in tests]) > 0
        except AssertionError:
            self._uvts_cls_logger.exception("Assertion exception occurred. No valid value for tests! "
                                            "Choose from ['kpss', 'adf', 'pp']. You can choose more than one.")
            sys.exit("STOP")

        do_test = list(compress(['kpss', 'adf', 'pp'], [i in ['kpss', 'adf', 'pp'] for i in tests]))
        return dict(
            zip(do_test, list(map(lambda x: ndiffs(self.ts_df['y'], test=x, alpha=alpha, max_d=max_d), do_test))))

    def acf_plots(self):
        """Generates autocorrelation plots"""
        fig, axes = plt.subplots(3, 2, figsize=(20, 9), sharex=False)
        #
        axes[0, 0].plot(self.ts_df['y'])
        axes[0, 0].set_title('Original Series')
        plot_acf(self.ts_df['y'], ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(self.ts_df['y'].diff())
        axes[1, 0].set_title('1st Order Differencing')
        plot_acf(self.ts_df['y'].diff().dropna(), ax=axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(self.ts_df['y'].diff().diff())
        axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(self.ts_df['y'].diff().diff().dropna(), ax=axes[2, 1])
        #
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def pacf_plots(self):
        """Generates partial correlation plots"""
        fig, axes = plt.subplots(3, 2, figsize=(20, 9), sharex=False)
        #
        axes[0, 0].plot(self.ts_df['y'])
        axes[0, 0].set_title('Original Series')
        plot_pacf(self.ts_df['y'], ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(self.ts_df['y'].diff())
        axes[1, 0].set_title('1st Order Differencing')
        # axes[0].set(ylim=(0, 5))
        plot_pacf(self.ts_df['y'].diff().dropna(), ax=axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(self.ts_df['y'].diff().diff())
        axes[2, 0].set_title('2nd Order Differencing')
        plot_pacf(self.ts_df['y'].diff().diff().dropna(), ax=axes[2, 1])

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    @abstractmethod
    def ts_fit(self, suppress=True):
        self.model_fit = None
        raise NotImplementedError("You must override ts_fit!")

    @abstractmethod
    def ts_test(self, show_plot=True):
        raise NotImplementedError("You must override ts_test!")

    def measure_rmse(self):
        """Computes root mean squared error on test data
        """
        try:
            assert self.residuals_forecast is not None
        except AssertionError:
            self._uvts_cls_logger.exception("AssertionError occurred, Cannot compute RMSE! Check your object mode")

        self.rmse = np.sqrt(sum(np.square(self.residuals_forecast)) / len(self.residuals_forecast))
        """
        if self._mode == 'test':
            self.rmse_test = self.rmse
        elif self._mode == 'test and validate':
            self.rmse_val = self.rmse - self.rmse_test
        elif self._mode == 'validate':
            self.rmse_val = self.rmse
        """

    def ts_validate(self, suppress=True, show_plot=True):
        """Validates the model"""
        if self._mode == 'forecast':  # or self._val_dt is None:
            self._uvts_cls_logger.warning("Nothing to validate! n_val not set within the initialization or you already "
                                          "used ts_forecast. In this case you have to restart and call ts_fit().")
            sys.exit("STOP")

        self._mode = 'test and validate'
        self.ts_fit(suppress=suppress)
        self.ts_test(show_plot=show_plot)

    def reset(self):
        for attr in self.__dict__.keys():
            setattr(self, attr, None)


# EoF class
def print_attributes(obj):
    """Prints attributes ob casted object obj"""
    for attr in obj.__dict__:
        if isinstance(getattr(obj, attr), pd.DataFrame) or isinstance(getattr(obj, attr), pd.Series):
            print(attr + " of shape: ", getattr(obj, attr).shape)
        else:
            print(attr, getattr(obj, attr))


def keys_f(keys):
    return [z for z in keys]