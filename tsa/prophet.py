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
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_cross_validation_metric
import holidays
from time import time


class ProphetForecaster(UVariateTimeSeriesClass):
    """Univariate time series child class using Prophet for forecasting,ref. to https://facebook.github.io/prophet

    Attributes
    ----------
    _prophet_interval_width: float
         The width of the uncertainty intervals (by default 80%), also
         ref. to https://facebook.github.io/prophet/docs/uncertainty_intervals.html
    _yearly_seasonality: bool
        Consider yearly seasonality yes/no
    _monthly_seasonality: bool
        Consider monthly seasonality yes/no
    _quarterly_seasonality: bool
       Consider quarterly seasonality yes/no
    _weekly_seasonality:
       Consider weekly seasonality yes/no
    _daily_seasonality: bool
       Consider daily seasonality yes/no
    _weekend_seasonality: bool#
       Consider week-end seasonality yes/no.
       ref. to https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events
    _changepoint_prior_scale: float
       If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility),
       you can adjust the strength of the sparse prior using this argument.
       By default, this parameter is set to 0.05. Increasing it will make the trend more flexible.
       Decreasing it will make the trend less flexible.
       ref. to https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet

    _changepoint_range: float
        By default changepoints are only inferred for the first 80% of the time series in order to have plenty of runway
        for projecting the trend forward and to avoid overfitting fluctuations at the end of the time series.
        This default works in many situations but not all, and can be changed using the changepoint_range argument.
        For example, m = Prophet(changepoint_range=0.9) will place potential changepoints in
        the first 90% of the time series.
        ref. to https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet
     _add_change_points: bool
        Whether to add change points to the plots
        ref. to https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet

     _diagnose: bool
         Whether to run cross validation yes/no
     _history: str
         Amount of historic data in days for cross validation,
         Corresponds to initial in  https://facebook.github.io/prophet/docs/diagnostics.html
     _step: str
         Correspons to period in the linke above. Defines step in days to shift the historic data
     _horizon: str
         Forecasting horizon in days for each cross validation run
    _consider_holidays: bool
         Whether to consider holiodays yes/no
         ref. to https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events
    _country: str
         The country for which holidays are to be considered

    _prophet_logger: Logger
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
       Diagnoses the fitted model. Cross validation is started
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
                 prophet_interval_width=0.95,
                 yearly_seasonality=False,
                 monthly_seasonality=False,
                 quarterly_seasonality=False,
                 weekly_seasonality=False,
                 daily_seasonality=False,
                 weekend_seasonality=False,
                 changepoint_prior_scale=0.001,
                 changepoint_range=0.9,
                 add_change_points=True,
                 diagnose=False,
                 history=None,
                 step=None,
                 horizon=None,
                 consider_holidays=True,
                 country='DE',
                 **kwds):
        """Initializes the object ProphetForecaster"""
        self._prophet_logger = Logger('prophet')

        try:
            super(ProphetForecaster, self).__init__(**kwds)
        except TypeError:
            self._prophet_logger.exception("TypeError occurred, Arguments missing")

        self._model = None

        self._prophet_interval_width = prophet_interval_width
        self._yearly_seasonality = yearly_seasonality
        self._monthly_seasonality = monthly_seasonality
        self._quarterly_seasonality = quarterly_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._daily_seasonality = daily_seasonality
        self._weekend_seasonality = weekend_seasonality

        self._changepoint_prior_scale = changepoint_prior_scale
        self._changepoint_range = changepoint_range
        self._add_change_points = add_change_points

        self._diagnose = diagnose
        self._history = history
        self._step = step
        self._horizon = horizon
        self._prophet_cv = None
        self._prophet_p = None

        self._consider_holidays = consider_holidays
        self._country = country

        self._id = 'Prophet'

    def __copy__(self):
        """Copies the object"""
        result = super(ProphetForecaster, self).__copy__()
        #
        result._model = self._model
        result._prophet_interval_width = self._prophet_interval_width
        result._yearly_seasonality = self._yearly_seasonality
        result._monthly_seasonality = self._monthly_seasonality
        result._quarterly_seasonality = self._quarterly_seasonality
        result._weekly_seasonality = self._weekly_seasonality
        result._daily_seasonality = self._daily_seasonality
        result._weekend_seasonality = self._weekend_seasonality

        result._changepoint_prior_scale = self._changepoint_prior_scale
        result._changepoint_range = self._changepoint_range
        result._add_change_points = self._add_change_points

        result._diagnose = self._diagnose
        result._history = self._history
        result._step = self._step
        result._horizon = self._horizon
        result._prophet_cv = self._prophet_cv
        result._prophet_p = self._prophet_p

        result._consider_holidays = self._consider_holidays
        result._country = self._country

        result._prophet_logger = self._prophet_logger

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
            elif k == "prophet_interval_width":
                self._prophet_interval_width = v
            elif k == "yearly_seasonality":
                self._yearly_seasonality = v
            elif k == "monthly_seasonality":
                self._monthly_seasonality = v
            elif k == "quarterly_seasonality":
                self._quarterly_seasonality = v
            elif k == "weekly_seasonality":
                self._weekly_seasonality = v
            elif k == "daily_seasonality":
                self._daily_seasonality = v
            elif k == "weekend_seasonality":
                self._weekend_seasonality = v
            elif k == "changepoint_prior_scale":
                self._changepoint_prior_scale = v
            elif k == "changepoint_range":
                self._changepoint_range = v
            elif k == "add_change_points":
                self._add_change_points = v
            elif k == "diagnose":
                self._diagnose = v
            elif k == "history":
                self._history = v
            elif k == "step":
                self._step = v
            elif k == "horizon":
                self._horizon = v
            elif k == "consider_holidays":
                self._consider_holidays = v
            elif k == "country":
                self._country = v

        return self

    def get_params_dict(self):
        """Gets parameters as a dictionary"""
        return {'prophet_interval_width': self._prophet_interval_width,
                'yearly_seasonality': self._yearly_seasonality,
                'monthly_seasonality': self._monthly_seasonality,
                'quarterly_seasonality': self._quarterly_seasonality,
                'weekly_seasonality': self._weekly_seasonality,
                'daily_seasonality': self._daily_seasonality,
                'weekend_seasonality': self._weekend_seasonality,
                'changepoint_prior_scale': self._changepoint_prior_scale,
                'changepoint_range': self._changepoint_range,
                'add_change_points': self._add_change_points,
                'diagnose': self._diagnose,
                'history': self._history,
                'step': self._step,
                'horizon': self._horizon,
                'consider_holidays': self._consider_holidays,
                'country': self._country
                }

    @staticmethod
    def we_season(ds):
        """Lambda function to prepare weekend_seasonality for  Prophet"""
        date = pd.to_datetime(ds)
        return date.weekday() >= 5

    def ts_fit(self, suppress=False):
        """Fit Prophet to the time series data.

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
            self._prophet_logger.info("***** Starting grid search *****")
            self._gs = self._gs.grid_search(suppress=suppress, show_plot=False)
            #
            self.best_model = self._gs.best_model
            self.__dict__.update(self.best_model['forecaster'].__dict__)
            self._prophet_logger.info("***** Finished grid search *****")
        else:
            self._prepare_fit()
            self._model = None
            self.ts_split()

            ts_df = self._train_dt.copy()
            ts_test_df = self._test_dt
            # sanity check
            if 'on_weekend' in ts_df.columns:
                ts_df.drop(['on_weekend', 'off_weekend'], inplace=True, axis=1)
                # ts_test_df.drop(['on_weekend', 'off_weekend'], inplace=True, axis=1)
            # Fit
            self._prophet_logger.info("Trying to fit the Prophet model....")
            try:
                if not suppress:
                    self._prophet_logger.info("...via using parameters\n")
                    print_attributes(self)
                # diagnose on?
                if self._diagnose:
                    try:
                        assert self._step is not None and self._horizon is not None
                    except (KeyError, AssertionError):
                        self._prophet_logger.warning("You want to diagnose the Prophet model. Please provide parameters "
                                                     "'step' and 'horizon' within object initialization!")
                        sys.exit("STOP")

                ts_df = ts_df.reset_index()
                ts_df.columns = self._ts_df_cols
                if ts_test_df is not None and not ts_test_df.empty:
                    ts_test_df = ts_test_df.reset_index()
                    ts_test_df.columns = self._ts_df_cols
                #
                weekly_s = self._weekly_seasonality
                if self._weekend_seasonality:
                    # force to False
                    weekly_s = False
                #
                if not self._consider_holidays:
                    self._model = Prophet(interval_width=self._prophet_interval_width,
                                          yearly_seasonality=self._yearly_seasonality,
                                          weekly_seasonality=weekly_s,
                                          daily_seasonality=self._daily_seasonality,
                                          changepoint_range=self._changepoint_range,
                                          changepoint_prior_scale=self._changepoint_prior_scale)
                else:
                    try:
                        assert self._country in ['AT', 'DE', 'US']
                    except AssertionError:
                        self._prophet_logger.exception("Assrtion exception occurred. Right now, Austria (AT), "
                                                       "Germany(DE) and USA (US) supported.")
                        sys.exit("STOP")
                    else:
                        holi = None
                        if self._country == 'AT':
                            holi = holidays.AT(state=None, years=list(np.unique(np.asarray(self.ts_df.index.year))))
                        elif self._country == 'DE':
                            holi = holidays.DE(state=None, years=list(np.unique(np.asarray(self.ts_df.index.year))))
                        elif self._country == 'US':
                            holi = holidays.US(state=None, years=list(np.unique(np.asarray(self.ts_df.index.year))))
                        #
                        holi_dict = dict()
                        for date, name in sorted(holi.items()):
                            holi_dict[date] = name

                        df_holi = pd.DataFrame.from_dict(data=holi_dict, orient='index').reset_index()
                        df_holi.columns = ['ds', 'holiday']
                        df_holi['lower_window'] = 0
                        df_holi['upper_window'] = 0
                        self._model = Prophet(interval_width=self._prophet_interval_width,
                                              yearly_seasonality=self._yearly_seasonality,
                                              weekly_seasonality=weekly_s,
                                              daily_seasonality=self._daily_seasonality,
                                              changepoint_range=self._changepoint_range,
                                              changepoint_prior_scale=self._changepoint_prior_scale,
                                              holidays=df_holi)

                if self._monthly_seasonality:
                    self._model.add_seasonality(name='monthly', period=30.5, fourier_order=20)
                    if not suppress:
                        self._prophet_logger.info("Added monthly seasonality.")

                if self._quarterly_seasonality:
                    self._model.add_seasonality(name='quarterly', period=91.5, fourier_order=20)
                    if not suppress:
                        self._prophet_logger.info("Added quarterly seasonality.")

                if self._weekend_seasonality:
                    ts_df['on_weekend'] = ts_df['ds'].apply(self.we_season)
                    ts_df['off_weekend'] = ~ts_df['ds'].apply(self.we_season)
                    self._train_dt = ts_df.copy()
                    self._train_dt.set_index('ds', inplace=True)
                    #
                    if ts_test_df is not None and not ts_test_df.empty:
                        ts_test_df['on_weekend'] = ts_test_df['ds'].apply(self.we_season)
                        ts_test_df['off_weekend'] = ~ts_test_df['ds'].apply(self.we_season)
                        self._test_dt = ts_test_df.copy()
                        self._test_dt.set_index('ds', inplace=True)
                    # and add
                    self._model.add_seasonality(name='weekend_on_season', period=7,
                                                fourier_order=5, condition_name='on_weekend')
                    self._model.add_seasonality(name='weekend_off_season', period=7,
                                                fourier_order=5, condition_name='off_weekend')

                    if not suppress:
                        self._prophet_logger.info("Added week-end seasonality.")

                # tic
                start = time()
                self.model_fit = self._model.fit(ts_df)
                # toc
                if not suppress:
                    self._prophet_logger.info("Time elapsed: {} sec.".format(time() - start))
            except (Exception, ValueError):
                self._prophet_logger.exception("Prophet error...")
                return -1
            else:
                self._prophet_logger.info("Model successfully fitted to the data!")

                # Fitted values
                self._prophet_logger.info("Computing fitted values and residuals...")
                # in-sample predict
                try:
                    self.fittedvalues = self._model.predict(ts_df.drop('y', axis=1))
                except (Exception, ValueError):
                    self._prophet_logger.exception("Prophet predict error...")
                # Residuals
                try:
                    # use fittedvalues to fill in the model dictionary
                    self.residuals = pd.Series(np.asarray(ts_df.y) - np.asarray(self.fittedvalues['yhat']),
                                               index=self._train_dt.index)
                except (KeyError, AttributeError):
                    self._prophet_logger.exception("Model was not fitted or ts has other structure...")
                #
                self.lower_conf_int = pd.Series(np.asarray(self.fittedvalues['yhat_lower']), index=self._train_dt.index)
                self.upper_conf_int = pd.Series(np.asarray(self.fittedvalues['yhat_upper']), index=self._train_dt.index)

                self._prophet_logger.info("Done.")
        return self

    def ts_diagnose(self):
        """Diagnoses the fitted model"""
        try:
            assert self.model_fit is not None
        except AssertionError:
            self._prophet_logger.exception("Model has to be fitted first! Please call ts_fit(...)")
            sys.exit("STOP")

        self.plot_residuals()

        if self._diagnose:
            if input("Run cross validation y/n? Note, depending on parameters provided "
                     "this can take some time...").strip().lower() == 'y':
                start = time()
                self._prophet_logger.info("Running cross validation using parameters provided....")
                if self._history is not None:
                    try:
                        self._prophet_cv = cross_validation(self.model_fit, initial=self._history,
                                                            period=self._step,
                                                            horizon=self._horizon)
                    except Exception:
                        self._prophet_logger.exception("Prophet cross validation error: check your "
                                                       "parameters 'history', 'horizon', 'step'!")
                else:
                    try:
                        self._prophet_cv = cross_validation(self.model_fit, period=self._step,
                                                            horizon=self._horizon)
                    except Exception:
                        self._prophet_logger.exception("Prophet cross validation error: "
                                                       "check your parameters 'horizon', 'step'!")

                self._prophet_logger.info("Time elapsed: {}".format(time() - start))
                simu_intervals = self._prophet_cv.groupby('cutoff')['ds'].agg(
                    [('forecast_start', 'min'),
                     ('forecast_till', 'max')])
                self._prophet_logger.info("Following time windows and cutoffs have been set-up:\n")
                print(simu_intervals)
                #
                plot_cross_validation_metric(self._prophet_cv, metric='mape')
                #
                self._prophet_logger.info("Running performance metrics...")
                self._prophet_p = performance_metrics(self._prophet_cv)

            else:
                self._prophet_logger.info("OK")
                return

    def plot_residuals(self):
        """Plot the residuals"""
        fig, axes = super(ProphetForecaster, self)._plot_residuals(
            y=np.asarray(self._train_dt['y']), yhat=np.asarray(self.fittedvalues['yhat']), _id="Prophet")
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        if super(ProphetForecaster, self)._check_ts_test() < 0:
            return

        self._prophet_logger.info("Evaluating the fitted Prophet model on the test data...")
        self.forecast = self._model.predict(self._test_dt.copy().reset_index().drop('y', axis=1))
        # confidence intervals
        self.lower_conf_int = pd.concat([self.lower_conf_int,
                                         pd.Series(np.asarray(self.forecast['yhat_lower']), index=self._test_dt.index)],
                                        axis=0)
        self.upper_conf_int = pd.concat([self.upper_conf_int,
                                         pd.Series(np.asarray(self.forecast['yhat_upper']), index=self._test_dt.index)],
                                        axis=0)

        self.residuals_forecast = pd.Series(np.asarray(self._test_dt['y']) - np.asarray(self.forecast['yhat']),
                                            index=self._test_dt.index)
        self.measure_rmse()
        self._prophet_logger.info("RMSE on test data: {}".format(self.rmse))
        # plot
        if show_plot:
            self.plot_forecast()

    def ts_forecast(self, n_forecast, suppress):
        """Forecast time series over time frame in the future specified via n_forecast"""
        #
        n_forecast = super(ProphetForecaster, self)._check_ts_forecast(n_forecast)
        #
        self._prophet_logger.info("Fitting using all data....")
        self._mode = 'forecast'
        self.ts_fit(suppress=suppress)

        self._prophet_logger.info("Forecasting next " + str(n_forecast) + str(self.ts_df.index.freq))
        #
        future = self._model.make_future_dataframe(periods=n_forecast, freq=self.freq)
        if self._weekend_seasonality:
            future['on_weekend'] = future['ds'].apply(self.we_season)
            future['off_weekend'] = ~future['ds'].apply(self.we_season)

        self.forecast = self._model.predict(future)
        # confidence intervals
        self.lower_conf_int = pd.concat([self.lower_conf_int,
                                         pd.Series(np.asarray(self.forecast['yhat_lower']), index=future.ds)],
                                        axis=0)
        self.upper_conf_int = pd.concat([self.upper_conf_int,
                                         pd.Series(np.asarray(self.forecast['yhat_upper']), index=future.ds)],
                                        axis=0)

        self.residuals_forecast = None
        self.plot_forecast()

    def plot_forecast(self):
        """Plot forecasted values"""
        if self.residuals_forecast is not None:
            fig, axes = super(ProphetForecaster, self)._plot_forecast(y=np.asarray(self._train_dt['y']),
                                                                      yhat=np.asarray(self.fittedvalues['yhat']),
                                                                      forecast=pd.Series(
                                                                          np.asarray(self.forecast['yhat']),
                                                                          index=self.forecast['ds']), _id='Prophet')
        else:
            fig_forecast = self._model.plot(self.forecast)
            fig_components = self._model.plot_components(self.forecast)
            if self._add_change_points:
                a = add_changepoints_to_plot(fig_forecast.gca(), self._model, self.forecast)

        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()
