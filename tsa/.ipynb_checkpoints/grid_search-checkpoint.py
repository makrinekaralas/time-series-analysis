__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

from tsa import Logger

from tsa import UVariateTimeSeriesForecaster
from tsa import ProphetForecaster
from tsa import DLMForecaster
from tsa import ARIMAForecaster
from tsa import SARIMAForecaster
from tsa import LinearForecaster
from tsa import ExponentialSmoothingForecaster

import sys
import numpy as np
from time import time
import itertools


class GridSearchClass(object):
    """Class to perform the grid search given the hyper parameters

    Attributes
    ----------
    forecaster: Object (tsa)
        Forecaster object from the tsa package
    hyper_params: dictionary
        A dictionary of hyper parameters
    results: dictionary
        A dictionary where results are saved
    best_model: dictionary
        A dictionary where the best model and respective hyper parameters are saved
    _gs_logger: Logger
        The logger for logging

    Methods
    ----------
    assertions()
       Assertion tests
    set_forecaster()
       Sets new forecaster
    set_hyper_params()
       Sets new hyper parameters
    grid_search()
       Performs grid search trough all combinations of parameters.
       Parameter combinations are generated using hyper parameters
    """
    def __init__(self, **kwargs):
        """Initializes GridSearch class"""
        self._gs_logger = Logger("grid_search")

        self.forecaster = None
        self.hyper_params = None

        for k, v in kwargs.items():
            if k == 'forecaster':
                self.forecaster = v
            elif k == 'hyper_params':
                self.hyper_params = v

        self.assertions()
        self.results = list()
        self.best_model = dict()
        #
        # self._gs_logger.info("Grid Search initialized. Call grid_search()")

    def assertions(self):
        if self.forecaster is not None:
            try:
                assert (isinstance(self.forecaster, ProphetForecaster) or isinstance(self.forecaster, DLMForecaster) \
                       or isinstance(self.forecaster, LinearForecaster) or \
                       isinstance(self.forecaster, ExponentialSmoothingForecaster) or \
                       isinstance(self.forecaster, ARIMAForecaster) or isinstance(self.forecaster, SARIMAForecaster)) \
                and not isinstance(self.forecaster, UVariateTimeSeriesForecaster)
            except AssertionError:
                self._gs_logger.exception("Unexpected type for forecaster!")
                sys.exit("STOP")

        if self.hyper_params is not None:
            try:
                assert isinstance(self.hyper_params, dict)
            except AssertionError:
                self._gs_logger.exception("Unexpected type for hyper_params")
                sys.exit("STOP")
        if hasattr(self.forecaster, 'n_test'):
            try:
                assert self.forecaster.n_test > 0
            except AssertionError:
                self._gs_logger.exception("No test data specified for this forecaster. Grid search will stop!")
                sys.exit("STOP")
            else:
                self.forecaster._mode = 'test'

    def set_forecaster(self, forecaster_obj):
        """Sets the forecaster"""
        self.forecaster = forecaster_obj
        self.assertions()

        return self

    def set_hyper_params(self, hyper_params):
        """Sets hyper parameters"""
        self.hyper_params = hyper_params
        self.assertions()

        return self

    @staticmethod
    def _print_dict(d):
        d_info = ""
        for k,v in d.items():
            d_info = d_info + "....................... | grid_search | INFO : " + str(k) + " : " + str(v) + "\n"
        return "Hyper parameter set: \n" + d_info

    def grid_search(self, suppress=False, show_plot=True):
        """Performs the grid search

        Via generating all possible combinations of parameters. The combinations are derived from the hyper parameters.
        This method assumes that attributes of a forecaster start with '_'
        The best model is chosen using rmse computed on the test data as
        the measure for the goodness of the forecaster
        """
        # set-up parameter sets
        for p, v in self.hyper_params.items():
            if not isinstance(v, list):
                self.hyper_params[p] = [v]

        combinations = list(itertools.product(*list(self.hyper_params.values())))
        params = [dict(zip(list(self.hyper_params.keys()), combinations[i])) for i in range(len(combinations))]
        self._gs_logger.info("{} number of parameter combinations generated".format(len(params)))
        #if input("Run grid search y/n?").strip().lower() == 'y':
        # reset
        self.results = list()
        self.best_model = dict()
        rmse = np.float('Inf')

        for i in range(len(params)):
            self._gs_logger.info(self._print_dict(params[i]))
            for p, val in params[i].items():
                # check
                attr = '_'+str(p)
                if attr in list(self.forecaster.__dict__.keys()):
                    _type = type(getattr(self.forecaster, attr))
                    try:
                        assert type(val) == _type
                    except AssertionError:
                        try:
                            if str(_type) == 'float':
                                val = np.float(val)
                            elif str(_type) == 'int':
                                val = np.int(val)
                            elif str(_type) == 'bool':
                                val = np.bool(val)
                            elif str(_type) == 'str':
                                val = str(val)
                            elif str(_type) == 'NoneType':
                                pass

                            self._gs_logger.info("Parameter type mismatch found, however, conversion successful")
                        except ValueError:
                            self._gs_logger.exception("Parameter type mismatch: Conversion did not work, "
                                                      "please check your hyper parameters!")
                            raise

                    setattr(self.forecaster, attr, val)
                else:
                    self._gs_logger.warning("Attribute {} not found. Default value will be used only.".format(attr))
                    pass

            # call ts_fit() and ts_test()
            # tic
            start = time()
            self.forecaster.ts_fit(suppress=suppress)
            self.forecaster.ts_test(show_plot=show_plot)
            # toc
            time_elapsed = time() - start
            #
            current_results = dict()
            current_results['params'] = self.forecaster.get_params_dict()
            current_results['rmse'] = self.forecaster.rmse
            current_results['time_elapsed'] = time_elapsed
            self.results.append(current_results)
            #
            if self.results[i]['rmse'] < rmse:
                rmse = self.results[i]['rmse']
                self.best_model['forecaster'] = self.forecaster.__copy__()
                self.best_model['hyper_params'] = self.results[i]['params']
                self.best_model['rmse'] = self.results[i]['rmse']
                self.best_model['time_elapsed'] = self.results[i]['time_elapsed']

        self._gs_logger.info("Best parameter combination:")
        self._gs_logger.info(self._print_dict(self.best_model['hyper_params']))
        self._gs_logger.info("RMSE {} :".format(self.best_model['rmse']))
       # else:
       #     self._gs_logger.info("OK")

        return self
