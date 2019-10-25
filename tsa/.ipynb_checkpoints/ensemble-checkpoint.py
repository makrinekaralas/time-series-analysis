__author__ = "Erik Pfeiffenberger"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

import sys
import numpy as np
import pandas as pd
from tsa import Logger
from tsa import ProphetForecaster
from tsa import ARIMAForecaster
from tsa import SARIMAForecaster
from tsa import AutoARIMAForecaster
from tsa import ExponentialSmoothingForecaster
from tsa import LinearForecaster
from tsa import DLMForecaster
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from random import sample


class EnsembleForecaster(LinearForecaster, SARIMAForecaster, ExponentialSmoothingForecaster,
                         AutoARIMAForecaster, ProphetForecaster, DLMForecaster):
    """Univariate time series class inheriting from all existing forecasters and choosing the best ensemble.

    Each forecaster is supposed to be equipped with y set of hyper parameters. Grid search is used to choose the
    best model for each forecaster among the respective hyper parameters.

    Those best models are then combined (Ensemble is created) to achieve the forecast of the best quality:
    All combinations of best models are created, forecasted values for all these combinations are either
    averaged or a median is computed. The best combination is chosen as the best ensemble.
    Note, that it is necessary that test and validation data are generated.

     Attributes
    ----------
    _model_list: list
       Internal (immutable) list of possible models
    ensemble: list
       List of all forecasters to be used to create the best ensemble.
    _dict_models: dictionary
       Dictionary keeping all models
    dict_hyper_params: dictionary
       Dictionary of hyper parameters per forecaster
    show_plots: bool
       Whether to show plots when models are fitted/tested
    _best_models: dictionary
       Dictionary keeping best models for each forecaster type in the list 'ensemble'.
       This best one is chosen after applying the grid search.
    best_ensemble: dictionary
        Dictionary keeping the results of ensemble
    _ensemble_logger: Logger
        The logger for logging

    Methods
    ----------
    assertions()
       Assertion tests, must be overrided
    ts_fit()
       Grid search on all forecasters in ensemble. Respective hyper parameters are used.
    ts_test()
       Test all forecasters on test data and computes rmse
    ts_validate()
       Validate all forecasters on validation data
    _build_ensemble()
       Builds the ensemble. All combinations of forecasters in ensemble is generated.
       For each combination the meand and median rmse over the validation data is computed.
       The best combination in terms of the best rmse is the best ensemble.

    """

    def __init__(self, dict_hyper_params, ensemble=['dlm', 'prophet'], show_plots=True, **kwds):
        """Initialized the object EnsembleForecaster"""
        self._model_list = ['arima', 'sarima', 'exponential smoothing', 'prophet', 'dlm', 'linear']

        self.ensemble = list(map(lambda x: x.lower(), ensemble))
        self.dict_hyper_params = dict_hyper_params
        self.show_plots = show_plots
        self._dict_models = dict()  # dict.fromkeys(self.ensemble, None)
        self._best_models = dict()
        self.best_ensemble = dict()
        self._ensemble_logger = Logger("ensemble")

        try:
            super(EnsembleForecaster, self).__init__(**kwds)
        except (TypeError, AttributeError) as e:
            self._ensemble_logger.exception("Arguments missing...{}".format(e))

        self._id = 'Ensemble'
        #
        if 'prophet' in self.ensemble:
            self._dict_models['prophet'] = self.__copy__()
            self._dict_models['prophet'].__class__ = ProphetForecaster
        if 'linear' in self.ensemble:
            self._dict_models['linear'] = self.__copy__()
            self._dict_models['linear'].__class__ = LinearForecaster
        if 'arima' in self.ensemble:
            self._dict_models['arima'] = self.__copy__()
            self._dict_models['arima'].__class__ = ARIMAForecaster
        if 'sarima' in self.ensemble:
            self._dict_models['sarima'] = self.__copy__()
            self._dict_models['sarima'].__class__ = SARIMAForecaster
        if 'exponential smoothing' in self.ensemble:
            self._dict_models['expsm'] = self.__copy__()
            self._dict_models['expsm'].__class__ = ExponentialSmoothingForecaster
        if 'dlm' in self.ensemble:
            self._dict_models['dlm'] = self.__copy__()
            self._dict_models['dlm'].__class__ = DLMForecaster
        if 'auto_arima' in self.ensemble:
            self._dict_models['auto_arima'] = self.__copy__()
            self._dict_models['auto_arima'].__class__ = AutoARIMAForecaster

        if 'all' in self.ensemble:
            self._dict_models['prophet'] = self.__copy__()
            self._dict_models['prophet'].__class__ = ProphetForecaster

            self._dict_models['linear'] = self.__copy__()
            self._dict_models['linear'].__class__ = LinearForecaster

            self._dict_models['arima'] = self.__copy__()
            self._dict_models['arima'].__class__ = ARIMAForecaster

            self._dict_models['sarima'] = self.__copy__()
            self._dict_models['sarima'].__class__ = SARIMAForecaster

            self._dict_models['expsm'] = self.__copy__()
            self._dict_models['expsm'].__class__ = ExponentialSmoothingForecaster

            self._dict_models['dlm'] = self.__copy__()
            self._dict_models['dlm'].__class__ = DLMForecaster

            self._dict_models['auto_arima'] = self.__copy__()
            self._dict_models['auto_arima'].__class__ = AutoARIMAForecaster

        self.assertions()

    def assertions(self):
        try:
            assert isinstance(self.dict_hyper_params, dict)
        except AssertionError:
            self._ensemble_logger.exception("Assertion exception occurred, dict expected")
            sys.exit("STOP")
        #
        """
        len_keys = list(filter(lambda x: x in list(self.dict_hyper_params.keys()),
                               keys_f(keys=self.ensemble)))
        try:
            assert len(len_keys) == len(self.ensemble)
        except AssertionError:
            self._dlm_logger.warning("hyper parameters found only for " + len_keys + " our of " + len(self.ensemble))
        """

        for k, v in self._dict_models.items():
            try:
                assert self._dict_models[k].n_test > 0 and self._dict_models[k].n_val > 0
            except AssertionError:
                self._ensemble_logger.exception("Assertion exception occurred,  both test and validation "
                                                "have to be generated! Please specify n_test and n_val!")
                sys.exit("STOP")

    def __copy__(self):
        """Copies the object"""

        result = super(EnsembleForecaster, self).__copy__()
        #
        result.ensemble = self.ensemble
        result.dict_hyper_params = self.dict_hyper_params
        result._dict_models = self._dict_models
        result._best_models = self._best_models
        result._ensemble_logger = self._ensemble_logger
        result._model_list = self._model_list

        return result

    def ts_fit(self, suppress=False):
        """Grid search on all forecasters in ensemble to find the best model out of hyper parameters provided.

         Parameters:
         ----------
         suppress: bool
            Suppress or not some of the output messages
         """
        for k, v in self._dict_models.items():
            if k in list(self.dict_hyper_params.keys()):
                self._gs.set_forecaster(self._dict_models[k])
                self._gs.set_hyper_params(self.dict_hyper_params[k])
                self._ensemble_logger.info(
                    "==========================================Starting grid search for the forecaster +++ {} +++ ==================================".format(
                        k))
                self._gs = self._gs.grid_search(suppress=suppress, show_plot=self.show_plots)
                #
                self._best_models[k] = self._gs.best_model
            else:
                self._dict_models[k].ts_fit(suppress=suppress)
                if k not in list(self._best_models.keys()):
                    self._best_models[k] = dict()
                    self._best_models[k]['forecaster'] = self._dict_models[k]

        return self

    def ts_test(self, show_plot=True):
        """Test all models on test data

         Parameters:
         ----------
         show_plot: bool
            Whether to show or not the residual plots
        """

        for k, v in self._best_models.items():
            self._ensemble_logger.info("==========================================Testing model +++ {} +++ ================================== ".format(k))
            if 'hyper_params' in list(self._best_models[k].keys()):
                self._best_models[k]['forecaster'].set_params(p_dict=self._best_models[k]['hyper_params'])
            self._best_models[k]['forecaster'].ts_test(show_plot=show_plot)

        self._build_ensemble()
        self._plot_ensemble()

    def plot_residuals(self):
        """Plot the residuals"""

        if self._best_models is None or len(self._best_models) == 0:
            self._ensemble_logger.warning("No models have been fit. The forecaster will stop!")
            sys.exit("STOP")

        for k, v in self._best_models.items():
            self._best_models[k]['forecaster'].plot_residuals()

    def ts_diagnose(self):
        """Plot the residuals"""

        if self._best_models is None or len(self._best_models) == 0:
            self._ensemble_logger.warning("No models have been fit. The forecaster will stop!")
            sys.exit("STOP")

        for k, v in self._best_models.items():
            self._best_models[k]['forecaster'].ts_diagnose()

    @staticmethod
    def _print_dict(d):
        e_info = ""
        for k, v in d.items():
            e_info = e_info + "....................... | ensemble | INFO : " + str(k) + " : " + str(v) + "\n"
        return "Best ensemble: \n" + e_info

    @staticmethod
    def lambda_forecast(x):
        if isinstance(x, ProphetForecaster):
            return x.forecast.iloc[:, -1].values
        else:
            return x.forecast.values

    def _compute_ensemble(self, compute_rmse=False):
        """Re-computes 'ensemble_forecast' for best_ensemble"""

        if self.best_ensemble['aggregation'] == 'none':
            self.best_ensemble['ensemble_forecast'] = pd.Series(self.lambda_forecast(self.best_ensemble['models'][0]),
                                                                index=self.best_ensemble['models'][0].forecast.index)
        elif self.best_ensemble['aggregation'] == 'mean':
            self.best_ensemble['ensemble_forecast'] = \
                pd.Series(np.mean(list(map(lambda x: self.lambda_forecast(x), self.best_ensemble['models'])), axis=0),
                          index=self.best_ensemble['models'][0].forecast.index)
            # rmse
            if compute_rmse:
                ensemble_res_mean = np.mean(list(map(lambda x: x.residuals_forecast, self.best_ensemble['models'])),
                                            axis=0)
                self.best_ensemble['rmse'] = np.sqrt(np.square(ensemble_res_mean)).mean()
        elif self.best_ensemble['aggregation'] == 'median':
            self.best_ensemble['ensemble_forecast'] = \
                pd.Series(np.median(list(map(lambda x: self.lambda_forecast(x), self.best_ensemble['models'])), axis=0),
                          index=self.best_ensemble['models'][0].forecast.index)
            if compute_rmse:
                ensemble_res_median = np.median(list(map(lambda x: x.residuals_forecast, self.best_ensemble['models'])),
                                                axis=0)
                self.best_ensemble['rmse'] = np.sqrt(np.square(ensemble_res_median)).mean()

    def _build_ensemble(self):
        """
        # check that validation has been run
        for k, v in self._best_models.items():
            if self._best_models[k]['forecaster']._mode != 'test and validate':
                # do what ts_validate does
                self._best_models[k]['forecaster'].set_params(p_dict=self._best_models[k]['hyper_params'])
                self._ensemble_logger.info("Validating model {}".format(k))
                self._best_models[k]['forecaster'].ts_validate(suppress=suppress, show_plot=show_plot)
            else:
                pass
        """
        # build ensemble
        self._ensemble_logger.info(
            "==========================================Start building the best ensemble==========================================")
        rmse = np.float('Inf')
        mod_list = list(self._best_models.keys())
        for L in range(0, len(mod_list) + 1):
            for subset in itertools.combinations(mod_list, L):
                if len(subset) == 0:
                    pass
                if len(subset) > 1:
                    #
                    ensemble_candidate = [self._best_models[s]['forecaster'] for s in subset]
                    # mean: note, residuals_forecast is now (each time) over the validation data
                    ensemble_res_mean = np.mean(list(map(lambda x: x.residuals_forecast, ensemble_candidate)),
                                                axis=0)
                    if np.sqrt(np.square(ensemble_res_mean)).mean() < rmse:
                        rmse = np.sqrt(np.square(ensemble_res_mean)).mean()
                        self.best_ensemble['rmse'] = rmse
                        self.best_ensemble['set'] = subset
                        self.best_ensemble['models'] = ensemble_candidate
                        self.best_ensemble['aggregation'] = 'mean'
                    # median
                    ensemble_res_median = np.median(list(map(lambda x: x.residuals_forecast, ensemble_candidate)),
                                                    axis=0)
                    if np.sqrt(np.square(ensemble_res_median)).mean() < rmse:
                        rmse = np.sqrt(np.square(ensemble_res_median)).mean()
                        self.best_ensemble['rmse'] = rmse
                        self.best_ensemble['set'] = subset
                        self.best_ensemble['models'] = ensemble_candidate
                        self.best_ensemble['aggregation'] = 'median'
                elif len(subset) == 1:
                    ensemble_candidate = self._best_models[subset[0]]['forecaster']
                    if ensemble_candidate.rmse < rmse:
                        rmse = ensemble_candidate.rmse
                        self.best_ensemble['rmse'] = rmse
                        self.best_ensemble['set'] = subset
                        self.best_ensemble['models'] = [ensemble_candidate]
                        self.best_ensemble['aggregation'] = 'none'
        self._compute_ensemble()

        # self._ensemble_logger.info("The best ensemble found as:")
        print(self._print_dict(self.best_ensemble))

    def _plot_ensemble(self):
        """Plots the best ensemble"""

        if len(self.best_ensemble) == 0:
            self._ensemble_logger.error("Ensemble does not exist yet! Forecaster will stop!")
            sys.exit("STOP")

        plt.figure(figsize=(20, 7))
        #
        plt.plot(self.best_ensemble['models'][0].ts_df, color='b')
        # colours
        colors = mcolors.BASE_COLORS
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         colo_name)
                        for colo_name, color in colors.items())
        colo_names = [name for hsv, name in by_hsv]
        if 'w' in colo_names:
            colo_names.remove('w')
        if 'b' in colo_names:
            colo_names.remove('b')
        if 'g' in colo_names:
            colo_names.remove('g')
        if 'darkgreen' in colo_names:
            colo_names.remove('darkgreen')
        colo_names = sample(colo_names, len(self.best_ensemble['models']))
        #
        for i in range(len(self.best_ensemble['models'])):
            plt.plot(pd.Series(self.lambda_forecast(self.best_ensemble['models'][i]),
                               index=self.best_ensemble['models'][i].forecast.index),
                     color=colo_names[i], linewidth=2.0,
                     label=str(type(self.best_ensemble['models'][i])).split("'")[1].split('.')[2])
        plt.plot(self.best_ensemble['ensemble_forecast'], color='darkgreen', linewidth=2.0, label='Ensemble')
        plt.axvline(x=min(self.best_ensemble['ensemble_forecast'].index), color='grey', linestyle='dashed')
        plt.legend()
        plt.title("Real (blue) and forecasted values")

        #
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.show()

    def ts_validate(self, suppress=True, show_plot=True):
        """Validate best ensemble."""

        if self.best_ensemble is None or len(self.best_ensemble) == 0:
            self._ensemble_logger.error("Ensemble has not been built! Forecaster will stop!")
            sys.exit("STOP")

        for i in range(len(self.best_ensemble['models'])):
            self.best_ensemble['models'][i]._mode = 'test and validate'
            self.best_ensemble['models'][i].ts_fit(suppress=suppress)
            self.best_ensemble['models'][i].ts_test(show_plot=show_plot)

        self._compute_ensemble(compute_rmse=True)
        print(self._print_dict(self.best_ensemble))
        self._plot_ensemble()

    def ts_forecast(self, n_forecast, features_dict=None, suppress=False):
        if self.best_ensemble is None or len(self.best_ensemble) == 0:
            self._ensemble_logger.error("Ensemble has not been built! Forecaster will stop!")
            sys.exit("STOP")

        for i in range(len(self.best_ensemble['models'])):
            if str(type(self.best_ensemble['models'][i])).split("'")[1].split('.')[2] != 'DLMForecaster':
                self.best_ensemble['models'][i].ts_forecast(n_forecast=n_forecast, suppress=suppress)
            else:
                self.best_ensemble['models'][i].ts_forecast(n_forecast=n_forecast, features_dict=features_dict,
                                                            suppress=suppress)

        self._compute_ensemble()
        self._plot_ensemble()

