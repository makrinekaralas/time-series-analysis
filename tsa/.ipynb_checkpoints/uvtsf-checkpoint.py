__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/

import sys

from tsa import Logger
from tsa import ProphetForecaster
from tsa import DLMForecaster
from tsa import ARIMAForecaster
from tsa import SARIMAForecaster
from tsa import AutoARIMAForecaster
from tsa import LinearForecaster
from tsa import ExponentialSmoothingForecaster


class UVariateTimeSeriesForecaster(LinearForecaster, AutoARIMAForecaster, SARIMAForecaster, ExponentialSmoothingForecaster,
                                   ProphetForecaster, DLMForecaster):
    """Univariate time series class inheriting from all existing forecasters and choosing the best forecaster.

     Attributes
    ----------
    forecasters: list
       List of all forecasters to be used. The best one will be chosen as the final model.
       The goodness of measure is rmse of a model on test data.
       Note, that it is necessary to generate the test data.

    _dict_models: dictionary
       Dictionary keeping all models
    best_model: Object
       The best model
    _uvtsf_logger: Logger
        The logger for logging

    Methods
    ----------
    assertions()
       Assertion tests, must be overrided
    ts_fit()
       Fits all forecasters to time series
    ts_test()
       Test all forecasters on test data and computes rmse
    _select_best()
       Helper function to select the best model
    select_best()
       Fits all forecasters to time series and selects the best one based on rmse of each computed on test data.
       If these was no test data, no test is done and no model is selected
    plot_residuals()
       Plots residuals for the best model
    ts_forecast()
       Forecasts time series and plots the results using the best model
    plot_forecasts()
       Plots forecasted time-series

    """
    def __init__(self, forecasters=['all'], **kwds):
        """Initialized the object UVariateTimeSeriesForecaster"""
        self.forecasters = list(map(lambda x: x.lower(), forecasters))
        self._dict_models = dict()  # .fromkeys(self._model_list, None)
        self.best_model = None

        self._uvtsf_logger = Logger("uvtsf")
        #
        try:
            super(UVariateTimeSeriesForecaster, self).__init__(**kwds)
        except TypeError:
            self._uvtsf_logger.exception("Arguments missing...")

        self._id = 'ts_forecaster'
        #
        if 'prophet' in self.forecasters:
            self._dict_models['prophet'] = self.__copy__()
            self._dict_models['prophet'].__class__ = ProphetForecaster
        if 'linear' in self.forecasters:
            self._dict_models['linear'] = self.__copy__()
            self._dict_models['linear'].__class__ = LinearForecaster
        if 'arima' in self.forecasters:
            self._dict_models['arima'] = self.__copy__()
            self._dict_models['arima'].__class__ = ARIMAForecaster
        if 'sarima' in self.forecasters:
            self._dict_models['sarima'] = self.__copy__()
            self._dict_models['sarima'].__class__ = SARIMAForecaster
        if 'auto_arima' in self.forecasters:
            self._dict_models['auto_arima'] = self.__copy__()
            self._dict_models['auto_arima'].__class__ = AutoARIMAForecaster
        if 'exponential smoothing' in self.forecasters:
            self._dict_models['expsm'] = self.__copy__()
            self._dict_models['expsm'].__class__ = ExponentialSmoothingForecaster
        if 'dlm' in self.forecasters:
            self._dict_models['dlm'] = self.__copy__()
            self._dict_models['dlm'].__class__ = DLMForecaster

        if 'all' in self.forecasters:
            self._dict_models['prophet'] = self.__copy__()
            self._dict_models['prophet'].__class__ = ProphetForecaster

            self._dict_models['linear'] = self.__copy__()
            self._dict_models['linear'].__class__ = LinearForecaster

            self._dict_models['arima'] = self.__copy__()
            self._dict_models['arima'].__class__ = ARIMAForecaster

            self._dict_models['sarima'] = self.__copy__()
            self._dict_models['sarima'].__class__ = SARIMAForecaster

            self._dict_models['auto_arima'] = self.__copy__()
            self._dict_models['auto_arima'].__class__ = SARIMAForecaster

            self._dict_models['expsm'] = self.__copy__()
            self._dict_models['expsm'].__class__ = ExponentialSmoothingForecaster

            self._dict_models['dlm'] = self.__copy__()
            self._dict_models['dlm'].__class__ = DLMForecaster

        self.assertions()
            
    def __copy__(self):
        """Copies the object"""

        result = super(UVariateTimeSeriesForecaster, self).__copy__()
        #
        result.forecasters = self.forecasters
        result._dict_models = self._dict_models
        result.best_model = self.best_model

        return result

    def assertions(self):
        try:
            assert isinstance(self.forecasters, list)
        except AssertionError:
            self._uvtsf_logger.exception("Assertion exception occurred, list expected for forecasters")
            sys.exit("STOP")

            
        for k, v in self._dict_models.items():
            try:
                assert self._dict_models[k].n_test > 0
            except AssertionError:
                self._uvtsf_logger.exception("Assertion exception occurred, no test data was generated! "
                                             "This forecaster requires the test data")
                sys.exit("STOP")

    def ts_fit(self, suppress=False):
        """Fit all forecasters to the time series data.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        for k,v in self._dict_models.items():
            if self._dict_models[k] is not None:
                self._dict_models[k].ts_fit(suppress=suppress)
        return self

    def ts_diagnose(self):
        """Diagnoses all candidate models"""
        for k,v in self._dict_models.items():
            if self._dict_models[k].model_fit is not None:
                self._dict_models[k].ts_diagnose()

    def ts_test(self, show_plot=True):
        """Test the fitted model if test data available"""
        for k, v in self._dict_models.items():
            if self._dict_models[k].model_fit is not None:
                self._dict_models[k].ts_test(show_plot=show_plot)
        self._select_best()

    def _select_best(self):
        """Helper function to select the best model among fitted forecasters"""
        rmse = float('Inf')
        for k, v in self._dict_models.items():
            if self._dict_models[k].model_fit is not None:
                if self._dict_models[k].rmse < rmse:
                    rmse = self._dict_models[k].rmse
                    self.best_model = self._dict_models[k]
        if self.best_model is not None:
            self._uvtsf_logger.info(
                "The best model selected as: {}".format(str(type(self.best_model)).split('\'')[1].split('.')[2]))
        else:
            self._uvtsf_logger.warning("No model has been fitted! Please call ts_fit()...")
    """
    def select_best(self, suppress=False):
        Fit all forecasters and select the best model
        self.ts_fit(suppress=suppress)
        self.ts_test()

        return self
    """

    def plot_residuals(self):
        """Residual plots"""
        if self.best_model is not None and bool(self.best_model):
            self.best_model.plot_residuals()
        else:
            for k, v in self._dict_models.items():
                if self._dict_models[k].model_fit is not None:
                    self._dict_models[k].plot_residuals()

    def ts_validate(self,  suppress=True, show_plot=True):
        """Validates the best model"""
        if self.best_model is not None:
            self.best_model.ts_validate(suppress=suppress, show_plot=show_plot)
        else:
            self._uvts_cls_logger.warning("No model has been selected yet! Run ts_test() first, or restart.")
            sys.exit("STOP")

    def ts_forecast(self, n_forecast, suppress=False):
        """Forecast n_forecast steps in the future using the best model"""
        if self.best_model is not None:
            self.best_model.ts_forecast(n_forecast=n_forecast, suppress=suppress)
        else:
            self._uvtsf_logger.warning("No model has been selected! Please call ts_test()...")
        return self    

    def plot_forecast(self):
        """Plots forecasted values"""
        if self.best_model is not None:
            self.best_model.plot_forecast()
        else:
            self._uvtsf_logger.warning("No model has been selected! Please call ts_fit()...")
