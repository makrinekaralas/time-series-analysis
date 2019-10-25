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
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from copy import copy, deepcopy
from scipy import stats


class UVariateTimeSeriesClass(object):
    """
    Uni-variate time series class
    Attributes:
        _ts_df_cols - internal column names for dataframe that will be input to model
        ts_df - time series data frame
        freq - frequency of time series, possibilities  ['S', 'min', 'H', 'D', 'W', 'M']
        p_train - float value defining which part of data is to be used as training data. Note, value of 1.0 would mean
                  all data will be used as training data,
                  hence no test data will be generated.
        timeformat - time format if time series data needs to be brought into datetime
        #
        _mode - defines the mode as 'test' or 'forecast'
        _train_dt - training data
        _test_dt - test data

        model_fit -  fitted model
        fittedvalues - computed fitted values
        residuals - residuals
        rmse - RMSE on test set (test data and the forecast on test data)

        upper_whisker_res - upper whisker for residuals
        lower_conf_int - upper confidence interval
        upper_conf_int - lower confidence interval

        forecast - computed forcatsed values
        residuals_forecast - residuals between forecasted and real values. Note, this variable exist only if test data
        existed

    Methods:
         ts_transform() - transforms time series using log10 or box-cox
         ts_resample() - resamples time series at the chosen frequency freq
         _plot_residuals() - residual plots helper function
         ts_test()  - evaluates fitted model on the test data, if this one has been generated
         ts_forecast() - forecasts time series and plots the results
         _plot_forecast() - helper function for plotting forecasted time-series
         ts_decompose() - decomposes time series in seasonal, trend and resduals and plots the results
         plot_decompose() - plots the results of ts_decompose()
    Helper methods:
         _prepare_fit() - prepares ts_fit of child class. Supposed to be called by a child class
         _residuals() - helper function for calculating residuals. Supposed to be called by a child class
         _check_ts_test() - checks for test. Supposed to be called by a child class
         _check_ts_forecast() - checks for forecast. Supposed to be called by a child class
    """

    def __init__(self, ts_df, time_format="%Y-%m-%d %H:%M:%S", freq='D', p_train=1.0, **kwds):
        """
        Initializes the object UVariateTimeSeriesForecaster
        """
        self._ts_df_cols = ['ds', 'y']

        self.ts_df = ts_df
        self.time_format = time_format
        self.freq = freq
        self.p_train = p_train
        self.transform = None
        self._boxcox_lmbda = None

        self._mode = ''

        self._train_dt = None
        self._test_dt = None

        self.model_fit = None
        self.fittedvalues = None
        self.residuals = None
        self.rmse = None

        self.upper_whisker_res = None
        self.lower_conf_int = None
        self.upper_conf_int = None

        self.forecast = None
        self.residuals_forecast = None

        self.seasonal = None
        self.trend = None
        self.baseline = None

        self._uvts_cls_logger = Logger('uvts_cls')
        # Assertion Tests
        try:
            assert self.freq in ['S', 'min', 'H', 'D', 'W', 'M']
        except AssertionError:
            self._uvts_cls_logger.warning("freq should be in  ['S', 'min', 'H', 'D', W', 'M']. "
                                          "Assuming daily frequency!")
            self.freq = 'D'

        try:
            self.p_train = float(self.p_train)
            assert self.p_train > 0
        except AssertionError:
            self._uvts_cls_logger.error("p_train defines part of data on which you would train your model."
                                        "This value cannot be less than or equal to zero!")
            self._uvts_cls_logger.exception("Exception occurred, p_train")
        except ValueError:
            self._uvts_cls_logger.error("p_train must be convertible to float type!")
            self._uvts_cls_logger.exception("Exception occurred, p_train")
        else:
            if int(self.p_train) < 1:
                self._mode = 'test'
            else:
                self._mode = 'forecast'

        try:
            assert pd.DataFrame(self.ts_df).shape[1] <= 2
        except AssertionError:
            self._uvts_cls_logger.error(
                "Time series must be uni-variate. "
                "Hence, at most a time columns and a column of numeric values are expected!")
            self._uvts_cls_logger.exception("Exception occurred, ts_df")
        else:
            self.ts_df = self.ts_df.reset_index()
            self.ts_df.columns = self._ts_df_cols
            self.ts_df['y'] = self.ts_df['y'].apply(np.float64, errors='coerce')
            self.ts_df.set_index('ds', inplace=True)
            print(type(self._uvts_cls_logger))
            print(self._uvts_cls_logger)
            self._uvts_cls_logger.info("Using time series data of range: " + str(min(self.ts_df.index)) + ' - ' + str(
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
        # delegate
        super(UVariateTimeSeriesClass, self).__init__(**kwds)

    def __copy__(self):
        """
        Copies the object
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """
        Deepcopies the object
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def ts_transform(self, transform):
        """
        Transforms time series via applying casted 'transform'. Right now 'log10' and 'box-cox' possible.
        """
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
                                                    "Possibly negative values present or bad lmbda?")
        return self

    def set_frequency(self, new_freq):
        """
        Sets new frequency and resamples time series to that new frequency
        """
        try:
            assert new_freq in ['S', 'min', 'H', 'D', 'W', 'M']
        except AssertionError:
            self._uvts_cls_logger.error("frequency should be in  ['S', 'min', 'H', 'D', W', 'M']")
        else:
            self.freq = new_freq
            self.ts_resample()

    def ts_check_frequency(self):
        """
        Checks the frequency of time series
        """
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
        """
        Brings original time series to the chosen frequency freq
        """
        ts_freq = pd.DataFrame(
            index=pd.date_range(self.ts_df.index[0], self.ts_df.index[len(self.ts_df) - 1], freq=self.freq),
            columns=['dummy'])
        self.ts_df = ts_freq.join(self.ts_df).drop(['dummy'], axis=1)
        self.ts_df.y = self.ts_df.y.fillna(method='ffill')
        # if np.isnan ( self.ts_df.y ).any ():
        #    self.ts_df.y = self.ts_df.y.fillna ( method='bfill' )
        if np.isnan(self.ts_df.y).any():
            self._uvts_cls_logger.warning("Some NaN found, something went wrong, check the data!")
            sys.exit(-1)

        self._uvts_cls_logger.info("Time series resampled at frequency: " + str(self.ts_df.index.freq) +
                                   ". New shape of the data: " + str(self.ts_df.shape))
        return self

    def _prepare_fit(self):
        """
        Prepares data for training or forecasting modes
        """

        if self.ts_df.index.freq is None:
            self._uvts_cls_logger.warning("Time series exhibit no frequency. Calling ts_resample()...")
            try:
                self.ts_resample()
            except ValueError:
                self._uvts_cls_logger.error("Resample did not work! Error:" + str(sys.exc_info()[0]))
                sys.exit("STOP")

        ts_df = self.ts_df
        ts_test_df = pd.DataFrame()

        if self._mode == 'forecast' or int(self.p_train) == 1:
            self._train_dt = ts_df
            self._test_dt = ts_test_df
        elif self._mode == 'test' and int(self.p_train) < 1:
            # split
            ts_df = ts_df.reset_index()
            ts_df.columns = self._ts_df_cols
            ts_test_df = ts_df
            # training
            ts_df = pd.DataFrame(ts_df.loc[:int(self.p_train * len(ts_df) - 1), ])
            ts_df.set_index('ds', inplace=True)
            # test
            ts_test_df = pd.DataFrame(ts_test_df.loc[int(self.p_train * len(ts_test_df)):, ])
            ts_test_df.set_index('ds', inplace=True)
            # now set
            self._train_dt = ts_df
            if not ts_test_df.empty:
                self._test_dt = ts_test_df

        return self

    def _residuals(self):
        """
        Calculate residuals
        """
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
        """
        Plot the residuals
        """
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
        if self.forecast is not None and self.residuals_forecast is None \
                and self.lower_conf_int is not None and self.upper_conf_int is not None:
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
        """
        Check before ts_test is child class is called
        """
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._uvts_cls_logger.exception("Model has to be fitted first! Please call ts_fit(...)")

        try:
            assert self._test_dt is not None
        except(KeyError, AssertionError):
            self._uvts_cls_logger.exception("Nothing to validate. "
                                            "Call ts_forecast() or specify amount of training data "
                                            "when initializing the object.")
            return -1
        else:
            self._mode = 'test'
            return 0

    def _check_ts_forecast(self, n_forecast):
        """
        Check before ts_forecast in child class is called
        """
        #
        try:
            n_forecast = int(n_forecast)
            assert 0 < n_forecast < len(self._train_dt)
        except AssertionError:
            self._uvts_cls_logger.exception("Number of periods to be forecasted is too low, too high or not numeric!")
        except ValueError:
            self._uvts_cls_logger.exception("n_forecast must be convertible to int type!")

        try:
            assert self.model_fit is not None
        except AssertionError:
            self._uvts_cls_logger.exception("Model has to be fitted first! Please call ts_fit(...)")

        return n_forecast

    def _gen_idx_future(self, n_forecast):
        idx_future = None
        if self.freq == 'S':
            idx_future = pd.date_range(start=max(self._train_dt.index),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           seconds=n_forecast - 1), freq='S')
        elif self.freq == 'min':
            idx_future = pd.date_range(start=max(self._train_dt.index),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           minutes=n_forecast - 1), freq='min')
        elif self.freq == 'H':
            idx_future = pd.date_range(start=max(self._train_dt.index),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           hours=n_forecast - 1), freq='H')
        elif self.freq == 'D':
            idx_future = pd.date_range(start=max(self._train_dt.index),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           days=n_forecast - 1), freq='D')
        elif self.freq == 'W':
            idx_future = pd.date_range(start=max(self._train_dt.index),
                                       end=max(self._train_dt.index) + datetime.timedelta(
                                           weeks=n_forecast - 1), freq='W')
        elif self.freq == 'M':
            idx_future = pd.date_range(start=max(self._train_dt.index),
                                       end=max(self._train_dt.index) + relativedelta(months=+(n_forecast - 1)),
                                       freq='M')
        return idx_future

    def _plot_forecast(self, y, yhat, forecast, _id):
        """
        Plot forecasted values
        """
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
        axes[0].plot(pd.Series(y, index=self._train_dt.index), color='b')
        #
        if self.residuals_forecast is not None:
            axes[0].plot(self.ts_df, color='b')
        axes[0].plot(forecast, color='darkgreen')
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
        """
        Decomposes time series into trend, seasonal and residual
        """
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

        if 'freq' not in list(params.keys()):
            params['freq'] = 1

        try:
            if self.ts_df.index.freq is not None:
                res = seasonal_decompose(self.ts_df.loc[:, 'y'], model=params['model'])
            else:
                res = seasonal_decompose(self.ts_df.loc[:, 'y'], model=params['model'], freq=params['freq'])

        except ValueError:
            self._uvts_cls_logger.exception("ValueError, seasonal_decompose error")
        else:
            self.seasonal = res.seasonal
            self.trend = res.trend
            self.baseline = self.seasonal + self.trend
            self.residuals = res.resid
            self.upper_whisker_res = self.residuals.mean() + 1.5 * (
                    self.residuals.quantile(0.75) - self.residuals.quantile(0.25))

    def plot_decompose(self):
        try:
            assert self.seasonal is not None
        except AssertionError:
            self.ts_decompose()

        fig, axes = plt.subplots(4, 1, figsize=(20, 7), sharex=True)
        axes[0].plot(self.trend)
        axes[0].set_title("Trend")
        #
        axes[1].plot(self.seasonal)
        axes[1].set_title("Seasonality")
        #
        axes[2].plot(self.baseline)
        axes[2].set_title("Baseline")
        #
        axes[3].plot(self.residuals)
        axes[3].set_title("Residuals")
        #
        if self.upper_whisker_res is not None:
            axes[3].axhline(y=self.upper_whisker_res,
                            xmin=0,
                            xmax=1, color='m',
                            label='upper_whisker',
                            linestyle='--', linewidth=1.5)
            axes[3].axhline(y=-self.upper_whisker_res,
                            xmin=0,
                            xmax=1, color='m',
                            label='upper_whisker',
                            linestyle='--', linewidth=1.5)

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_fit(self):
        # stop the delegation chain
        assert not hasattr(super(), 'ts_fit')

    # root mean squared error or rmse
    def measure_rmse(self):
        try:
            assert self.residuals_forecast is not None
        except AssertionError:
            self._uvts_cls_logger.exception("AssertionError occurred, Cannot compute RMSE! Check your object mode")

        self.rmse = np.sqrt(np.square(self.residuals_forecast).mean())


# EoF class
def print_attributes(obj):
    """
    Prints attributes ob catsed object obj
    """
    for attr in obj.__dict__:
        if isinstance(getattr(obj, attr), pd.DataFrame):
            print(attr + " of shape: ", getattr(obj, attr).shape)
        else:
            print(attr, getattr(obj, attr))
