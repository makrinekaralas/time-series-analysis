__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

from tsa import Logger
from tsa import UVariateTimeSeriesClass
from tsa import print_attributes
from tsa import keys_f

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydlm import dlm, trend, seasonality, dynamic, autoReg, longSeason
from time import time


class DLMForecaster(UVariateTimeSeriesClass):
    """Univariate time series child class using DLM of pydlm for forecasting,
    ref. to https://pydlm.github.io/pydlm_user_guide.html

    Attributes
    ----------
    _dlm_trend: tuple
         A tuple of degree, discount, name and prior covariance
    _dlm_seasonality: tuple
        A tuple of period, discount, name and prior covariance
    _dlm_dynamic: dictionary
        A dictionary of tuples as features, discount, name and prior covariance.
        Note, the features for _dynamic should be a list of lists.
    _dlm_auto_reg: tuple
       A tuple of degree, discount, name and prior covariance
    _dlm_long_season: tuple
       A tuple of period, stay, name and prior covariance
    _use_rolling_window: bool
       Use rolling window in forward filtering yes/no
    _window_size: int
    _dlm_interval_width: float
       TBD
    _dlm_logger: Logger
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
    plot_dlm()
        Plot pydlm native plots
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
                 dlm_trend=None,
                 dlm_seasonality=None,
                 dlm_dynamic=None,
                 dlm_auto_reg=None,
                 dlm_long_season=None,
                 use_rolling_window=False,
                 window_size=0,
                 dlm_interval_width=0.95,
                 **kwds):
        """Initializes the object DLMForecaster"""
        if dlm_trend is None:
            dlm_trend = {'degree': 0, 'discount': 0.99, 'name': 'trend1', 'w': 1e7}
        self._model = None
        self.mse = None

        self._dlm_trend = dlm_trend
        self._dlm_seasonality = dlm_seasonality
        self._dlm_dynamic = dlm_dynamic
        self._dlm_auto_reg = dlm_auto_reg
        self._dlm_long_season = dlm_long_season
        self._use_rolling_window = use_rolling_window
        self._window_size = window_size
        self._dlm_interval_width = dlm_interval_width

        self._dlm_logger = Logger('dlm')

        self.assertions()

        try:
            super(DLMForecaster, self).__init__(**kwds)
        except TypeError:
            self._dlm_logger.exception("TypeError occurred, Arguments missing")

        self._id = 'DLM'
        self._train_dlm_dynamic = None  # features
        self._test_dlm_dynamic = None  # featureDict
        self._val_dlm_dynamic = None   # featureDict

    def assertions(self):
        if self._dlm_trend is not None:
            try:
                assert isinstance(self._dlm_trend, dict)
            except AssertionError:
                self._dlm_logger.exception("Assertion exception occurred, dictionary expected for dlm_trend")
                sys.exit("STOP")
            else:
                len_keys = list(filter(lambda x: x in list(self._dlm_trend.keys()),
                                       keys_f(keys=['degree', 'discount', 'name'])))
                try:
                    assert len(len_keys) == len(['degree', 'discount', 'name'])
                except AssertionError:
                    self._dlm_logger.exception("Not all expected parameters found for trend. "
                                               "['degree', 'discount', 'name'] are necessary!")
                    sys.exit("STOP")
                else:
                    if 'w' not in list(self._dlm_trend.keys()):
                        self._dlm_trend['w'] = 1e7

        if self._dlm_seasonality is not None:
            try:
                assert isinstance(self._dlm_seasonality, dict)
            except AssertionError:
                self._dlm_logger.exception("Assertion exception occurred, dictionary expected for dlm_seasonality")
                sys.exit("STOP")
            else:
                len_keys = list(filter(lambda x: x in list(self._dlm_seasonality.keys()),
                                       keys_f(keys=['period', 'discount', 'name'])))
                try:
                    assert len(len_keys) == len(['period', 'discount', 'name'])
                except AssertionError:
                    self._dlm_logger.exception("Not all expected parameters found for seasonality. "
                                               "['period', 'discount', 'name] are necessary!")
                    sys.exit("STOP")
                else:
                    if 'w' not in list(self._dlm_seasonality.keys()):
                        self._dlm_seasonality['w'] = 1e7

        if self._dlm_auto_reg is not None:
            try:
                assert isinstance(self._dlm_auto_reg, dict)
            except AssertionError:
                self._dlm_logger.exception("Assertion exception occurred, dictionary expected for dlm_auroReg")
                sys.exit("STOP")
            else:
                len_keys = list(filter(lambda x: x in list(self._dlm_auto_reg.keys()),
                                       keys_f(keys=['degree', 'discount', 'name'])))
                try:
                    assert len(len_keys) == len(['degree', 'discount', 'name'])
                except AssertionError:
                    self._dlm_logger.exception("Not all expected parameters found for auto_reg. "
                                               "['degree', 'discount', 'name'] are necessary!")
                    sys.exit("STOP")
                else:
                    if 'w' not in list(self._dlm_auto_reg.keys()):
                        self._dlm_auto_reg['w'] = 1e7

        if self._dlm_long_season is not None:
            try:
                assert isinstance(self._dlm_long_season, dict)
            except AssertionError:
                self._dlm_logger.exception("Assertion exception occurred, dictionary expected for dlm_longSeason")
                sys.exit("STOP")
            else:
                len_keys = list(filter(lambda x: x in list(self._dlm_long_season.keys()),
                                       keys_f(keys=['period', 'stay', 'name'])))
                try:
                    assert len(len_keys) == len(['period', 'stay', 'name'])
                except AssertionError:
                    self._dlm_logger.exception("Not all expected parameters found for long season. "
                                               "['period', 'stay', 'name'] are necessary!")
                    sys.exit("STOP")
                else:
                    if 'w' not in list(self._dlm_long_season.keys()):
                        self._dlm_long_season['w'] = 1e7

        if self._dlm_dynamic is not None:
            try:
                assert isinstance(self._dlm_dynamic, dict)
            except AssertionError:
                self._dlm_logger.exception("Assertion exception occurred, dictionary expected for dlm_seasonality")
                sys.exit("STOP")
            else:
                try:
                    assert 'features' in list(self._dlm_dynamic.keys())
                except AssertionError:
                    self._dlm_logger.exception("Assertion exception occurred, 'features' must be provided!")
                    sys.exit("STOP")
                else:
                    try:
                        assert isinstance(self._dlm_dynamic['features'], list)
                    except AssertionError:
                        self._dlm_logger.exception("Assertion exception occurred, list expected for 'features'")
                        sys.exit("STOP")
                    else:
                        for i in range(len(self._dlm_dynamic['features'])):
                            len_keys = list(filter(lambda x: x in list(self._dlm_dynamic['features'][i].keys()),
                                                   keys_f(keys=['features', 'discount', 'name'])))
                            try:
                                assert len(len_keys) == len(['features', 'discount', 'name'])
                            except AssertionError:
                                self._dlm_logger.exception("Not all expected parameters found for dynamic features. "
                                                           "['features', 'discount', 'name'] are necessary!")
                                sys.exit("STOP")

                    # features must have same length with the data
                    for i in range(len(self._dlm_dynamic['features'])):
                        try:
                            assert len(self._dlm_dynamic['features'][i]['features']) == len(self.ts_df)
                        except AssertionError:
                            self._dlm_logger.exception("Assertion exception occurred. All provided features must"
                                                       " be of same length as your data!")
                            sys.exit("STOP")
                        else:
                            if 'w' not in list(self._dlm_dynamic['features'][i].keys()):
                                self._dlm_dynamic['features'][i]['w'] = 1e7

        if self._use_rolling_window:
            try:
                assert self._window_size > 0
            except AssertionError:
                self._dlm_logger.exception("Assertion exception occurred, zero window_size. "
                                           "No rolling window will be used")
                self._use_rolling_window = False

    def __copy__(self):
        """Copies the object"""
        result = super(DLMForecaster, self).__copy__()
        #
        result._dlm_trend = self._dlm_trend
        result._dlm_seasonality = self._dlm_seasonality
        result._dlm_dynamic = self._dlm_dynamic
        result._dlm_auto_reg = self._dlm_auto_reg
        result._dlm_long_season = self._dlm_long_season
        result._use_rolling_window = self._use_rolling_window
        result._window_size = self._window_size
        result._dlm_interval_width = self._dlm_interval_width

        result._dlm_logger = self._dlm_logger

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
            elif k == "dlm_trend":
                self._dlm_trend = v
            elif k == "dlm_seasonality":
                self._dlm_seasonality = v
            elif k == "dlm_dynamic":
                self._dlm_dynamic = v
            elif k == "dlm_autoReg":
                self._dlm_auto_reg = v
            elif k == "dlm_longSeason":
                self._dlm_long_season = v
            # TBD other params!!    
        self.assertions()

        return self

    def get_params_dict(self):
        """Gets parameters as dictionary"""
        return {'dlm_trend': self._dlm_trend,
                'dlm_seasonality': self._dlm_seasonality,
                'dlm_dynamic': self._dlm_dynamic,
                'dlm_auto_reg': self._dlm_auto_reg,
                'dlm_long_season': self._dlm_long_season,
                'use_rolling_window': self._use_rolling_window,
                'window_size': self._window_size, 
                'dlm_interval_width': self._dlm_interval_width
                }

    def ts_split(self):
        """DLM extension of the parent ts_split()

        DLM needs to extend the ts_split of its parent class.
        The reason lies in dynamic features: this list of lists must be splitted
        """
        # call super
        super(DLMForecaster, self).ts_split()

        if self._dlm_dynamic is None or self._mode == 'forecast':
            return self

        # split dynamic features
        test_feat_dict = dict()
        val_feat_dict = dict()

        self._train_dlm_dynamic = self._dlm_dynamic

        for i in range(len(self._dlm_dynamic)):
            feats = self._dlm_dynamic['features'][i]['features']
            #
            if self._mode == 'test and validate':
                if self._test_dlm_dynamic is not None:
                    self._train_dlm_dynamic['features'][i]['features'].append(
                        self._test_dlm_dynamic[self._dlm_dynamic['features'][i]['name']])
                    self._val_dlm_dynamic = self._test_dlm_dynamic
                else:
                    self._dlm_logger.error("Something is wrong, mode!")
            else:
                if self._mode == 'test' and self.n_val == 0:
                    self._train_dlm_dynamic['features'][i]['features'] = feats[:(len(feats) - 1 - self.n_test)]
                    #
                    test_feat_dict[self._dlm_dynamic['features'][i]['name']] = feats[(len(feats) - self.n_test):]
                elif self._mode == 'validate':
                    self._train_dlm_dynamic['features'][i]['features'] = feats[:(len(feats) - 1 - self.n_val)]
                    #
                    val_feat_dict[self._dlm_dynamic['features'][i]['name']] = feats[(len(feats) - self.n_val):]
                elif self._mode == 'test' and self.n_val > 0:
                    self._dlm_dynamic['features'][i]['features'] = feats[:(len(feats) - 1 - self.n_test - self.n_val)]
                    #
                    test_feat_dict[self._dlm_dynamic['features'][i]['name']] = \
                        feats[(len(feats) - self.n_test - self.n_val):(len(feats) - self.n_val - 1)]
                    val_feat_dict[self._dlm_dynamic['features'][i]['name']] = feats[(len(feats) - self.n_val):]

            # now set
            if len(test_feat_dict):
                self._test_dlm_dynamic = test_feat_dict
            if len(val_feat_dict):
                self._val_dlm_dynamic = val_feat_dict

        return self

    def ts_fit(self, suppress=False):
        """Fit DLM to the time series data.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        if self.hyper_params is not None:
            self._gs.set_forecaster(self)
            self._gs.set_hyper_params(self.hyper_params)
            # a very important command here to avoid endless loop
            self.hyper_params = None
            self._dlm_logger.info("***** Starting grid search *****")
            self._gs = self._gs.grid_search(suppress=suppress, show_plot=False)
            #
            self.best_model = self._gs.best_model
            self.__dict__.update(self.best_model['forecaster'].__dict__)
            self._dlm_logger.info("***** Finished grid search *****")
        else:
            self._prepare_fit()
            self._model = None
            self.ts_split()

            ts_df = self._train_dt.copy()

            # Fit
            self._dlm_logger.info("Trying to fit the DLM model....")
            try:
                if not suppress:
                    self._dlm_logger.info("...via using parameters\n")
                    print_attributes(self)

                ts_df = ts_df.reset_index()
                ts_df.columns = self._ts_df_cols

                self._model = dlm(ts_df['y'])

                # trend
                if self._dlm_trend is not None:
                    self._model = self._model + trend(degree=self._dlm_trend['degree'], discount=self._dlm_trend['discount'],
                                                      name=self._dlm_trend['name'], w=self._dlm_trend['w'])
                # seasonality
                if self._dlm_seasonality is not None:
                    self._model = self._model + seasonality(period=self._dlm_seasonality['period'],
                                                            discount=self._dlm_seasonality['discount'],
                                                            name=self._dlm_seasonality['name'], w=self._dlm_seasonality['w'])
                # dynamic
                if self._train_dlm_dynamic is not None:
                    for i in range(len(self._train_dlm_dynamic['features'])):
                        self._model = self._model + dynamic(features=self._train_dlm_dynamic['features'][i]['features'],
                                                            discount=self._train_dlm_dynamic['features'][i]['discount'],
                                                            name=self._train_dlm_dynamic['features'][i]['name'],
                                                            w=self._train_dlm_dynamic['features'][i]['w'])
                # auto_reg
                if self._dlm_auto_reg is not None:
                    self._model = self._model + autoReg(degree=self._dlm_auto_reg['degree'], discount=self._dlm_auto_reg['discount'],
                                                        name=self._dlm_auto_reg['name'], w=self._dlm_auto_reg['w'])
                # long_season
                if self._dlm_long_season is not None:
                    ls = longSeason(period=self._dlm_long_season['period'], stay=self._dlm_long_season['stay'], data=ts_df,
                                    name=self._dlm_long_season['name'], w=self._dlm_long_season['w'])
                    self._model = self._model + ls

                if not suppress:
                    self._dlm_logger.info("The constructed DLM model components:")
                    print(self._model.ls())

                # tic
                start = time()
                if self._use_rolling_window:
                    self._model.fitForwardFilter(useRollingWindow=True, windowLength=self._window_size)
                    self._model.fitBackwardSmoother()
                else:
                    self._model.fit()
                self.model_fit = self._model
                # toc
                if not suppress:
                    self._dlm_logger.info("Time elapsed: {} sec.".format(time() - start))
            except (Exception, ValueError) as e:
                self._dlm_logger.exception("DLM error...{}".format(e))
                return -1
            else:
                self._dlm_logger.info("Model successfully fitted to the data!")
                self._dlm_logger.info("Computing fitted values and residuals...")

                # Residuals
                self.residuals = pd.Series(self.model_fit.getResidual(),
                                           index=self._train_dt.index)
                try:
                    self.lower_conf_int = pd.Series(self.model_fit.getInterval()[1], index=self._train_dt.index)
                    self.upper_conf_int = pd.Series(self.model_fit.getInterval()[0], index=self._train_dt.index)
                except ValueError as e:
                    self._dlm_logger.exception("Something went wrong in getInterval...{}".format(e))

                self.mse = self.model_fit.getMSE()

                # Fitted values
                # this is not elegant, but found no other way
                self.fittedvalues = self._train_dt['y'] + self.residuals

        return self

    def ts_diagnose(self):
        """Diagnoses the fitted model"""
        self.plot_residuals()

    def plot_dlm(self):
        """Plot pydlm native plots"""
        self.model_fit.plot("DLM native")

    def plot_residuals(self):
        """Plot the residuals."""
        fig, axis = super(DLMForecaster, self)._plot_residuals(y=np.asarray(self._train_dt['y']),
                                                               yhat=np.asarray(self.fittedvalues),
                                                               _id="DLM")

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        if super(DLMForecaster, self)._check_ts_test() < 0:
            return

        N = len(self._test_dt)

        self._dlm_logger.info("Evaluating the fitted DLM model on the test data...")

        if self._test_dlm_dynamic is not None:
            (predictMean, predictVar) = self._model.predictN(N=N, date=self._model.n-1,
                                                             featureDict=self._test_dlm_dynamic)
        else:
            (predictMean, predictVar) = self._model.predictN(N=N, date=self._model.n-1)

        self.forecast = pd.Series(np.asarray(predictMean), index=self._test_dt.index)
        # confidence intervals
        cl, cu = self.compute_ci(yhat=np.asarray(predictMean), yhat_var=np.asarray(predictVar),
                                 ci_level=self._dlm_interval_width)
        cl = pd.Series(cl, index=self._test_dt.index)
        cu = pd.Series(cu, index=self._test_dt.index)
        self.lower_conf_int = pd.concat([self.lower_conf_int, cl], axis=0)
        self.upper_conf_int = pd.concat([self.upper_conf_int, cu], axis=0)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt['y']) - np.asarray(self.forecast),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._dlm_logger.info("RMSE on test data: {}".format(self.rmse))
        # plot
        if show_plot:
            self.plot_forecast()
        
        return self

    def ts_forecast(self, n_forecast, suppress=False, features_dict=None):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(DLMForecaster, self)._check_ts_forecast(n_forecast)
        #
        self._dlm_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._dlm_logger.info("Forecasting next " + str(n_forecast) + str(self.freq))
        #
        try:
            if features_dict is not None and len(features_dict) != 0:
                (predictMean, predictVar) = self._model.predictN(N=n_forecast, date=self._model.n - 1,
                                                                 featureDict=features_dict)
            else:
                (predictMean, predictVar) = self._model.predictN(N=n_forecast, date=self._model.n - 1)
        except (NameError, ValueError) as e:
            self._dlm_logger.exception("DLM PredictN error...{}".format(e))
            sys.exit("STOP")

        idx_future = self._gen_idx_future(n_forecast=n_forecast)
        self.forecast = pd.Series(np.asarray(predictMean), index=idx_future)
        
        # confidence intervals
        cl, cu = self.compute_ci(yhat=np.asarray(predictMean), yhat_var=np.asarray(predictVar),
                                 ci_level=self._dlm_interval_width)
        cl = pd.Series(cl, index=idx_future)
        cu = pd.Series(cu, index=idx_future)
        self.lower_conf_int = pd.concat([self.lower_conf_int, cl], axis=0)
        self.upper_conf_int = pd.concat([self.upper_conf_int, cu], axis=0)
        
        self.residuals_forecast = None
        self.plot_forecast(n_forecast=n_forecast, features_dict=features_dict)
        return self

    def plot_forecast(self, **kwargs):
        """Plot forecasted values"""
        if self.residuals_forecast is not None:
            fig, axis = super(DLMForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                  yhat=np.asarray(self.fittedvalues),
                                                                  forecast=self.forecast, _id='DLM')
            plt.gcf().autofmt_xdate()
            plt.grid(True)
            plt.show()
        else:
            n_forecast = -1
            features_dict = dict()

            for k, v in kwargs.items():
                if k == 'n_forecast':
                    n_forecast = v
                if k == 'features_dict':
                    features_dict = v
            print(features_dict)  
            try:
                if features_dict is not None and len(features_dict) != 0:
                    self._model.plotPredictN(N=n_forecast, date=self._model.n - 1, featureDict=features_dict)
                else:
                    self._model.plotPredictN(N=n_forecast, date=self._model.n - 1)
            except (NameError, ValueError) as e:
                self._dlm_logger.exception("DLM plotPredictN error...{}".format(e))
                sys.exit("STOP")