#!/usr/bin/env python.
__author__ = "Makine Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Makine Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"


import os
import sys
import numpy as np
import pandas as pd
import math
import datetime
from dateutil.relativedelta import relativedelta
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.gofplots import qqplot
from copy import copy, deepcopy
import holidays
from scipy import stats
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from time import time, sleep

   
class imdict(dict):
    """
    
    Implements immutable dictionary.
    """
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('This object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear       = _immutable
    update      = _immutable
    setdefault  = _immutable
    pop         = _immutable
    popitem     = _immutable

def we_season(ds):
    """
    Apply function to prepare weekend_seasonality for  Prophet
    """
    date = pd.to_datetime(ds)
    return ( date.weekday() >=5 )
    
class UVariateTimeSeriesForecaster ( object ):
    """ 
    
    Univariate time series forecasterg class
    Attributes:
        _ts_dict_keys - expected keys for the dictionary ts_dict
        _ts_df_cols - internal column names for dataframe that will be input to model
        _defparams_ses (immutable) - default parameters for simple exponential smoothing
        _defparams_es (immutable) - default parameters for exponential smoothing
        _defparams_auto_arima (immutable) - default parameters for auto_arima
         ts_df - time series data frame
        _freq - frequency of time series, possibilities  ['S', 'min', 'H', 'D', 'W', 'M']
        _p_train - float value defining which part of data is to be used as training data. Note, value of 1.0 would mean all data will be used.
        _timeformat - time format if time series data needs to be brought into datetime
         model - a dictionary containing eveyrthing around model
    Methods:
         ts_resample() - brings ts to the wished frequency freq
         ts_fit()      - fits the chosen model casted via method to the training data
         ts_diagnose() - plots diagnostics plots
         plot_residuals() - residual plots
         ts_test()  - evaluates fitted model on test data, if this one has been generated
         ts_forecast() - forecasts time series and plots the results
         plot_forecasts() - plots forecasted time-series
         ts_decompose() - time series decomposition in seasonal, trend and resduals
    """

    def __init__(self, ts_dict):
        """
        
        Initializes the object UVariateTimeSeriesForecaster
             ts_dict - a dictionary that initializes the class. It has to be of the structure
                       {'ts_df': ts_df, 'time_format': '%Y-%m-%d %H:%M:%S.%f', 'freq': freq, 'p_train': p_train} with
                       ts the time series dataframe
                       time_format - time format for converting time in that data frame (if needed so)
                       freq - the frequency of time series (resampling will be done by the class)
                       p_train - a float value defining a part of data to be used for training (if wished so). the rest of the data will be understood as test data.
        """
        # constants and parameters
        self._ts_dict_keys = ['ts_df', 'time_format', 'freq', 'transform', 'lmbda', 'p_train'] 
        self._ts_df_cols = ['ds', 'y']
        self._defparams_ses = imdict ( {
            'smoothing_level': None,
            'optimized': True } )
        self._defparams_es = imdict ( {
            'trend': 'additive',
            'seasonal': None,
            'seasonal_periods': 0,
            'damped': True,
            'smoothing_level': None,
            'smoothing_slope': None,
            'smoothing_seasonal': None,
            'damping_slope': None,
            'optimized': True,
            'use_boxcox': False,  # {True, False, ‘log’, float}
            'remove_bias': False,
            'use_brute': True } )
        self._defparams_auto_arima = imdict ( {
            'start_p': 1,
            'start_q': 1,
            'test': 'adf',  # use adftest to find optimal 'd'
            'max_p': 3,
            'max_q': 3,     # maximum p and q
            'm': 1,         # frequency of series
            'd': None,      # let model determine 'd'
            'seasonal': False,  # Allow/No Seasonality
            'D': None,      # let model determine 'D'
            'start_P': 1,
            'start_Q': 1,
            'max_P': 3,
            'max_Q': 3 })
        self._defparams_prophet = imdict ( {
            'interval_width': 0.95, 
            'yearly_seasonality': False, 
            'monthly_seasonality': False,
            'quarterly_seasonality': False,
            'weekly_seasonality': False, 
            'daily_seasonality': True,
            'weekend_seasonality': False,     # customized!
            'country': 'AT',
            'state': None,
            'consider_holidays': True,
            'changepoint_prior_scale': 0.001, #make Trend 'not flexible' 
            'changepoint_range': 0.9,
            'add_change_points': True,
            'diagnose': False,
            'period': None,
            'horizon': None
        } )
        self._defparams_linear = imdict ( {
            'fit_intercept': True, 
            'normalize': True, 
            'copy_X': False, 
            'n_jobs': None
        } )

        # Assertion Tests
        try:
            assert isinstance(ts_dict, dict)  
        except AssertionError:
            print (
                "A dictionary of form {'ts_df': ts_df, 'time_format': '%Y-%m-%d %H:%M:%S.%f', 'freq': freq, 'transform': transformation,  'p_train': p_train} is expected for this variable!" )
            sys.exit ( "STOP" )
        # check for keys
        main_keys = [ z for z in self._ts_dict_keys if not z in ['lmbda'] ]
        len_keys = list ( filter ( lambda x: x in list ( ts_dict.keys () ),   main_keys ) )
        try:
            assert len ( len_keys ) == len ( main_keys )
        except AssertionError:
            print ( "Not all keys in dictionary ts_dict could be located! Located only: {}".format ( len_keys ) )
            sys.exit ( "STOP" )
        # check for values
        # 1
        try:
            assert ts_dict['freq'] in ['S', 'min', 'H', 'D', 'W', 'M']
        except AssertionError:
            print ( "freq should be in  ['S', 'min', 'H', 'D', W', 'M']. Assuming hourly frequency!")
            self._freq = 'H'
        else:
            self._freq = ts_dict['freq']
        # 2
        try:
            assert ts_dict['transform'].lower().strip() in ['log10', 'box-cox', '']
        except AssertionError:
            print ( "transform should be in ['ln', 'box-cox'] or empty. Assuming no transform! Hence, if you get bad results, you would like maybe to choose e.g., log10 here.")
            self._transform = None
        else:
            self.transform =  ts_dict['transform'].lower()
        # 3
        try:
            self._p_train = float ( ts_dict['p_train'] )
            assert self._p_train > 0
        except AssertionError:
            print (
                "p_train defines part of data on which you would train your model. This value cannot be less than or equal to zero!" )
            sys.exit ( "STOP" )
        except ValueError:
            print ( "p_train must be convertable to float type!" )
            sys.exit ( "STOP" )
        # 4
        try:
            assert pd.DataFrame ( ts_dict['ts_df'] ).shape[1] <= 2
        except AssertionError:
            print ( "Time series must be univariate. Hence, at most time and value columns are expected!" )
            sys.exit ( "STOP" )
        else:
            self.ts_df = ts_dict['ts_df'].reset_index ()
            self.ts_df.columns = self._ts_df_cols
            self.ts_df['y'] = self.ts_df['y'].apply ( np.float64, errors='coerce' )
            #transform
            if self.transform == 'log10':
                print("Applying log10 transform.")
                try:
                    self.ts_df['y'] = self.ts_df['y'].apply ( np.log10 )
                except ValueError as e:
                    print("log10 transformation did not work! Negative values present?... {}".format(e))
                    sys.exit ( "STOP" )
            elif self.transform == 'box-cox':
                if 'lmbda' in list ( ts_dict.keys() ):
                    self._boxcox_lmbda = ts_dict['lmbda']
                else:
                    self._boxcox_lmbda = None
                try:
                    if self._boxcox_lmbda is None:
                        bc, lmbda_1 = stats.boxcox(self.ts_df['y'],lmbda=self._boxcox_lmbda) 
                        self.ts_df['y'] =  stats.boxcox(self.ts_df['y'],lmbda=lmbda_1) 
                    else:
                        self.ts_df['y'] =  stats.boxcox(self.ts_df['y'],lmbda=self._boxcox_lmbda) 
                        
                except ValueError as e:
                    print("box-cox transformation did not work! Negative values present or bad lmbda?... {}".format(e))
                    sys.exit ( "STOP" )
            elif self.transform.strip()=='':
                print("No transformation.")
                
            self.ts_df.set_index ( 'ds', inplace=True )
            print ( "Using time series data of range: " + str ( min ( self.ts_df.index ) ) + ' - ' + str ( max ( self.ts_df.index ) ) + " and shape: " + str (self.ts_df.shape ) )
        # 5
        if not isinstance ( self.ts_df.index, pd.DatetimeIndex ):
            print ( "Time convertion is required..." )
            self.ts_df = self.ts_df.reset_index ()
            try:
                self.ts_df['ds'] = self.ts_df['ds'].apply (
                    lambda x: datetime.datetime.strptime (
                        str ( x ).translate ( {ord ( 'T' ): ' ', ord ( 'Z' ): None} )[:-1],
                        ts_dict['time_format'] ) )
            except ValueError as e:
                print ( "Zulu time convertion not successfull: {}".format ( e ) )
                print ( "Will try without assuming zulu time..." )
                try:
                    self.ts_df['ds'] = self.ts_df['ds'].apply (
                        lambda x: datetime.datetime.strptime ( str ( x ), ts_dict['time_format'] ) )
                except ValueError as e:
                    print ( "Time convertion not successfull. Check your time_format: {}".format ( e ) )
                else:
                    print ( "Time convertion successfull!" )
                    self._timeformat = ts_dict['time_format']
            else:
                print ( "Time convertion successfull!" )
                self._timeformat = ts_dict['time_format']
            # set index                
            self.ts_df.set_index ( 'ds', inplace=True )
        else:
            self._timeformat = ts_dict['time_format']
        #    
        self.ts_df.index = pd.to_datetime ( self.ts_df.index )
        self.ts_df.sort_index ( inplace=True )
        #
        self.model_params = dict()
        self.model = dict ()
        self.model['method'] = ""
        self.model['m'] = None        #used for Prophet
        self.model['m_fit'] = None
        #
        print ( "The object is initialized!" )
        
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
    
    def set_model_params(self, dict_params):
        """
        
        Sets casted parameters as model parameters
        """
        self.model_params = dict_params 
        
    def set_frequency(self, new_freq):
        """
        
        Sets new frequency and resamples time series to that new frequency
        """
        try:
            assert new_freq in ['S', 'min', 'H', 'D', 'W', 'M']
        except AssertionError:
            print ( "frequency should be in  ['S', 'min', 'H', 'D', W', 'M']" )
            sys.exit( "STOP" )
        else:
            self._freq = new_freq
            self.ts_df_resample()
            
    def ts_check_frequency(self):
        """
        
        Checks the frequency of time series
        """
        if self.ts_df.index.freq is None:
            print ( "No specific frequency detected yet.\n" )
            print ( "Frequency chosen: " + str(self._freq) + " enter 'n' and call ts_resample() if you are satisfied with this value." )  
            if input ( "Should a histogram of time deltas be plotted y/n?" ).strip().lower() != 'y':                   
                sys.exit( "STOP" )
            else:       
                ff = pd.Series ( self.ts_df.index[1:(len ( self.ts_df ) ) ] - self.ts_df.index[0:(len ( self.ts_df ) - 1)] )
                ff = ff.apply (lambda x: int ( x.total_seconds() / ( 60*60 ) ) )
                plt.hist ( ff, bins = 120 )
                plt.xlabel ( "Rounded time delta [H]" )
                plt.ylabel ( "Frequency of occurance" )
                print(ff.value_counts())
                print ( "Should hourly frequency not fit, choose a reasonable frequency and call set_frequency(new_freq)" )
        else:
            print ( "Timse series frequency: " + str ( self.ts_df.index.freq ) )    
            
    def ts_resample(self):
        """ 
        
        Brings original time seiries to the chosen frequency freq
        """
        ts_freq = pd.DataFrame (
            index=pd.date_range ( self.ts_df.index[0], self.ts_df.index[len ( self.ts_df ) - 1], freq=self._freq ),
            columns=['dummy'] )
        self.ts_df = ts_freq.join ( self.ts_df ).drop ( ['dummy'], axis=1 )
        self.ts_df.y = self.ts_df.y.fillna ( method='ffill' )
        #if np.isnan ( self.ts_df.y ).any ():
        #    self.ts_df.y = self.ts_df.y.fillna ( method='bfill' )
        if np.isnan ( self.ts_df.y ).any ():    
            print ( "Some NaN found, something went wrong, check the data!" )
            sys.exit ( "STOP" )

        print ( "Timse series resampled at frequency: " + str ( self.ts_df.index.freq ) + ". New shape of data: " + str ( self.ts_df.shape ) )
       
    def ts_fit(self,
               method='auto_arima',
               rem_we=False, 
               dict_params=None,
              mode='train',
              suppress=False):
        """
        
        Fit a model defined by method to the time series data.
        Parameters:
            - method: defines method
            - rem_we: True/False if week end should be included/excluded from the data. Please note, the value True is not recommended 
                      since time-series data gets gaps! Use prophet instead and set True for weekend_seasonality. 
            - dict_params: dictionary of model parameters. If None default values will be used. If some keys not given, these will be filled via default values.
            - mode: training ('train') or forecasting ('forecast') mode. The difference is that in forecasting mode first all data is used to fit the model 
                    and the forecast is done afterwards, whereas in training mode only part of data (_p_train) is used to fit the model.
            - suppress: surpresses output messages. 
        """

        def i_fillparams(dict_params, def_params):
            """
            
            Fills into dictionary of parameters via their default values 
            in case some parameters are not specified
            """
            if dict_params is None or len(dict_params) == 0:
                if self.model_params is None or len(self.model_params) == 0:
                    def_values =  list ( def_params.keys () )
                    dict_params = dict()
                else:
                    def_values =  list ( filter (
                        lambda x: x not in list ( self.model_params.keys () ), list ( def_params.keys () ) ) )
                    dict_params = self.model_params
            else:
                def_values = list ( filter (
                    lambda x: x not in list ( dict_params.keys () ), list ( def_params.keys () ) ) )
                
            try:
                assert isinstance(dict_params, dict) 
            except AssertionError:
                print ( "Dictionary is expected for dict_params!" )
                
            if len ( def_values ):
                for k, v in def_params.items ():
                    if k in def_values:
                        dict_params[k] = v
            return dict_params

        try:
            assert method.lower () in ['lin', 'auto_arima', 'ses', 'es', 'prophet']
        except AssertionError:
            print ( "Method is out of the scope! Right now 'lin', auto_arima','ses' ,'es' or 'prophet' possible" )
            sys.exit( "STOP" )
        else:
            self.model['method'] = method.lower ()
            
        # prep time series
        if self.ts_df.index.freq is None:
            print ( "Time series exhibit no frequency yet. Calling ts_resample()..." )
            #
            try:
                self.ts_resample ()
            except:
                print ( "Resample did not work!" )
                print ( "Error:", sys.exc_info ()[0] )
                sys.exit ( "STOP" )

        ts_df = self.ts_df.copy()
        ts_test_df = pd.DataFrame ()
        #keep week-end? think twice since this changes data and time series gets gaps!
        if rem_we and not self.model['method'] in ['ses', 'es']:
            ts_df['weekday'] = ts_df.index.weekday.map ( str )
            ts_df['weekday'] = ts_df['weekday'].apply ( np.int_, errors='coerce' )
            ts_df = ts_df[ts_df['weekday'] < 5]
            ts_df = ts_df.drop ( ['weekday'], axis=1 )
            self.ts_df = ts_df
        elif rem_we and self.model['method'] in ['ses', 'es']:
            print (
                "Cannot remove week-end when using es or ses as method. Use lin or auto_arima to achieve this, however, we generally not recommend this! Now using data ASIS" )
        #    
        if int ( self._p_train ) == 1:
            mode = 'forecast'
            
        if mode == 'train':
            ts_df = ts_df.reset_index ()
            ts_df.columns = self._ts_df_cols
            ts_test_df = ts_df
            #training
            ts_df = pd.DataFrame ( ts_df.loc[:int ( self._p_train * len ( ts_df ) - 1 ), ] )
            ts_df.set_index ( 'ds', inplace=True )
            #test
            ts_test_df = pd.DataFrame ( ts_test_df.loc[int ( self._p_train * len ( ts_test_df ) ):, ] )
            ts_test_df.set_index ( 'ds', inplace=True )
        #    
        self.model['train_dt'] = ts_df
        if not ts_test_df.empty:
            self.model['test_dt'] = ts_test_df
        #
        if self.model['method'] == 'lin':
            """
            Fit LinearRegression
            """
            if not suppress:
                print ( "Trying to fit the linear model...." )
            try:
                dict_params = i_fillparams ( dict_params=dict_params,
                                         def_params=self._defparams_linear )
                self.model_params = dict_params
                if not suppress:
                    print("...via using parameters\n")
                    print(pd.DataFrame.from_dict(dict_params, orient='index'))
                #
                x = np.arange ( 0, len ( ts_df ) ).reshape ( -1, 1 )
                y = np.asarray ( ts_df['y'] ) 
                self.model['m_fit'] = LinearRegression ( fit_intercept=dict_params['fit_intercept'], normalize=dict_params['normalize'],
                                                         copy_X=dict_params['copy_X'], n_jobs=dict_params['n_jobs']).fit ( x , y )
            except (Exception, ValueError) as e:
                print ( "LinearRegression error...: {}".format ( e ) )
            else:
                if not suppress:
                    print ( "Model successfully fitted to the data!" )
                    print ( "R^2: {:f}".format ( self.model['m_fit'].score( x , y ) ) )
            
            self.model['intercept'] =  self.model['m_fit'].intercept_
            self.model['slope'] =  self.model['m_fit'].coef_
            if not suppress:
                print ( "The estimated linear slope: {}".format(self.model['slope'] ) )
            #
            #self.model['fittedvalues'] = pd.DataFrame ( (self.model['intercept'] * x + self.model['slope']).flatten(), index=ts_df.index )
            self.model['fittedvalues'] = pd.Series ( self.model['m_fit'].predict(x), index=ts_df.index )
            
        elif self.model['method'] == 'prophet':
            """
            Fit Prophet
            """
            print ( "Trying to fit the Prophet model...." )
            try:
                dict_params = i_fillparams ( dict_params=dict_params,
                                         def_params=self._defparams_prophet )
                self.model_params = dict_params
                print("...via using parameters\n")
                print(pd.DataFrame.from_dict(dict_params, orient='index'))
                #
                if self.model_params['diagnose']:
                        try:
                            assert not self.model_params['period'] is None and not self.model_params['horizon'] is None
                        except (KeyError, AssertionError):
                            print (
                                "You want to diagnose the Prophet model. Please provide parameters 'period' and 'horizon' withinb object initialization!" )
                            sys.exit ( "STOP " )
                
                ts_df = ts_df.reset_index ()
                ts_df.columns = self._ts_df_cols
                if not ts_test_df.empty:
                    ts_test_df = ts_test_df.reset_index ()
                    ts_test_df.columns = self._ts_df_cols
                #
                weekly_s = dict_params['weekly_seasonality']
                if dict_params['weekend_seasonality']:
                    weekly_s = False
                #    
                if not dict_params['consider_holidays']:
                    self.model['m'] = Prophet ( interval_width=dict_params['interval_width'],
                                                yearly_seasonality=dict_params['yearly_seasonality'],
                                                weekly_seasonality=weekly_s,
                                                daily_seasonality=dict_params['daily_seasonality'],
                                                changepoint_range=dict_params['changepoint_range'],
                                                changepoint_prior_scale=dict_params['changepoint_prior_scale'] )
                else:
                        try:
                            assert not dict_params['country'] is None and len ( dict_params['country'] ) != 0 and dict_params['country'] in [
                                'AT', 'DE', 'US']
                        except AssertionError:
                            print (
                                "You need to consider holidays? Then provide country name! Right now, Austria, Germany and US supported." )
                            sys.exit ( "STOP" )
                        else:
                            if dict_params['country'] == 'AT':
                                holi = holidays.AT ( state=None, years=list ( np.unique ( np.asarray ( self.ts_df.index.year ) ) ) )
                            elif dict_params['country'] == 'DE':
                                holi = holidays.DE ( state=None, years=list ( np.unique ( np.asarray ( self.ts_df.index.year ) ) ) )
                            elif dict_params['country'] == 'US':
                                holi = holidays.US ( state=None, years=list ( np.unique ( np.asarray ( self.ts_df.index.year ) ) ) )
                        #
                        holi_dict = dict()
                        for date, name in sorted ( holi.items () ):
                            holi_dict [ date ] = name
                            
                        df_holi = pd.DataFrame.from_dict(data = holi_dict, orient='index').reset_index()
                        df_holi.columns = [ 'ds','holiday' ]
                        df_holi['lower_window'] = 0
                        df_holi['upper_window'] = 0
                        print ( "Considering holidays for country {}:\n".format ( dict_params['country'] ) )
                        print ( df_holi )
                        self.model['m'] = Prophet ( interval_width=dict_params['interval_width'],
                                                    yearly_seasonality=dict_params['yearly_seasonality'],
                                                    weekly_seasonality=weekly_s,
                                                    daily_seasonality=dict_params['daily_seasonality'],
                                                    changepoint_range=dict_params['changepoint_range'],
                                                    changepoint_prior_scale=dict_params['changepoint_prior_scale'],
                                                    holidays=df_holi )
                if dict_params['monthly_seasonality']:
                    print ( "Adding monthly seasonality." )
                    self.model['m'].add_seasonality ( name='monthly', period=30.5,  fourier_order=20 )
                if dict_params['quarterly_seasonality']:
                    print ( "Adding quarterly seasonality." )
                    self.model['m'].add_seasonality ( name='quarterly', period=91.5,  fourier_order=20 )
                    
                if dict_params['weekend_seasonality']:
                    print ( "Adding week-end seasonality." )
                    ts_df['on_weekend'] = ts_df['ds'].apply ( we_season )
                    ts_df['off_weekend'] = ~ts_df['ds'].apply ( we_season )  
                    self.model['train_dt'] = ts_df.copy()
                    self.model['train_dt'].set_index ( 'ds', inplace=True )
                    #
                    if not ts_test_df.empty:
                        ts_test_df['on_weekend'] = ts_test_df['ds'].apply ( we_season )
                        ts_test_df['off_weekend'] = ~ts_test_df['ds'].apply ( we_season )                    
                        self.model['test_dt'] = ts_test_df.copy()
                        self.model['test_dt'].set_index ( 'ds', inplace=True )
                    #
                    self.model['m'].add_seasonality ( name='weekend_on_season', period=7,
                                                      fourier_order=5, condition_name='on_weekend' )
                    self.model['m'].add_seasonality ( name='weekend_off_season', period=7,
                                                      fourier_order=5, condition_name='off_weekend' )
                    
                #if input ( "Continue with fitting y/n?" ).strip().lower() != 'y':
                #    sys.exit( "STOP" )
                start = time ()
                self.model['m_fit'] = self.model['m'].fit ( ts_df )
                print ( "Time elapsed: {}".format ( time () - start ) )
                if dict_params['consider_holidays']:
                    print ( "Holidays included:\n" )
                    print ( self.model['m'].train_holiday_names )
            except (Exception, ValueError) as e:
                print ( "Prophet error...: {}".format ( e ) )
            else:
                print ( "Model successfully fitted to the data!" )
            #in-sample predict    
            try:    
                self.model['fittedvalues'] = self.model['m'].predict ( ts_df.drop ( 'y', axis=1 ) ) 
            except (Exception, ValueError) as e:
                print ( "Prophet predict error...: {}".format ( e ) )
        elif self.model['method'] == 'ses':
            """
            Fit SimpleExpSmoothing
            """
            dict_params = i_fillparams ( dict_params=dict_params,
                                         def_params=self._defparams_ses )
            self.model_params = dict_params
            
            print ( "Trying to fit the SimpeExpSmoothing model...." )
            try:
                self.model['m_fit'] = SimpleExpSmoothing ( ts_df ).fit ( smoothing_level=dict_params['smoothing_level'],
                                                                      optimized=dict_params['optimized'] )
            except (Exception, ValueError) as e:
                print ( "Simple Exponential Smoothing error...: {}".format ( e ) )
            else:
                print ( "Model successfully fitted to the data!" )
                self.model['fittedvalues'] = self.model['m_fit'].fittedvalues
        elif self.model['method'] == 'es':
            """
            Fit ExponentialSmoothing
            """
            dict_params = i_fillparams ( dict_params=dict_params,
                                         def_params=self._defparams_es )
            self.model_params = dict_params

            print ( "Trying to fit the ExponentialSmoothing model...." )
            try:
                self.model['m_fit'] = ExponentialSmoothing ( ts_df, freq=self._freq, trend=dict_params['trend'],
                                                             seasonal=dict_params['seasonal'],
                                                             damped=dict_params['damped'] ).fit (
                    smoothing_level=dict_params['smoothing_level'],
                    smoothing_slope=dict_params['smoothing_slope'],
                    smoothing_seasonal=dict_params['smoothing_seasonal'],
                    damping_slope=dict_params['damping_slope'],
                    optimized=dict_params['optimized'],
                    use_boxcox=dict_params['use_boxcox'],
                    remove_bias=dict_params['remove_bias']
                )
            except (Exception, ValueError) as e:
                print ( "Exponential Smoothing error...: {}".format ( e ) )
            else:
                print ( "Model successfully fitted to the data!" )
                self.model['fittedvalues'] = self.model['m_fit'].fittedvalues
        elif self.model['method'] == 'auto_arima':
            """
            Fit auto_arima
            """
            dict_params = i_fillparams ( dict_params=dict_params,
                                         def_params=self._defparams_auto_arima )
            self.model_params = dict_params

            print ( "Trying to fit the auto_arima model...." )
            print("...via using parameters\n")
            print ( pd.DataFrame.from_dict(dict_params, orient='index') )
            
            start = time()
            try:
                self.model['m_fit'] = pm.auto_arima ( ts_df, start_p=dict_params['start_p'],
                                                      start_q=dict_params['start_q'],
                                                      test=dict_params['test'],
                                                      max_p=dict_params['max_p'], max_q=dict_params['max_q'],
                                                      m=dict_params['m'],
                                                      d=dict_params['d'],
                                                      seasonal=dict_params['seasonal'],
                                                      D=dict_params['D'],
                                                      start_P=dict_params['start_P'], start_Q=dict_params['start_Q'],
                                                      max_P=dict_params['max_P'], max_Q=dict_params['max_Q'],
                                                      trace=True,
                                                      error_action='ignore',
                                                      suppress_warnings=True,
                                                      stepwise=True )
            except (Exception, ValueError) as e:
                print ( "Will try to reset some parameters...: {}".format ( e ) )
                try:
                    self.model['m_fit'] = pm.auto_arima ( ts_df, start_p=dict_params['start_p'],
                                                          start_q=dict_params['start_q'],
                                                          test=dict_params['test'],
                                                          max_p=dict_params['max_p'], max_q=dict_params['max_q'],
                                                          m=1,
                                                          d=0,
                                                          seasonal=dict_params['seasonal'],
                                                          D=0,
                                                          start_P=dict_params['start_P'],
                                                          start_Q=dict_params['start_Q'],
                                                          max_P=dict_params['max_P'], max_Q=dict_params['max_Q'],
                                                          trace=True,
                                                          error_action='ignore',
                                                          suppress_warnings=True,
                                                          stepwise=True )
                except (Exception, ValueError) as e:
                    print ( "Please try other parameters: {}!".format ( e ) )
                    self.model['m_fit'] = None
                    sys.exit ( "STOP" )
                except:
                    print ( "Error:", sys.exc_info ()[0] )
                    self.model['m_fit'] = None
                    sys.exit ( "STOP" )
            else:
                print("Time elapsed: {}".format ( time()-start ) )
                #
                print ( "Model successfully fitted to the data!" )
                print ( "The chosen model AIC: " + str ( self.model['m_fit'].aic () ) )
                #
                print(ts_df.index)
                self.model['fittedvalues'] = pd.Series (
                    self.model['m_fit'].predict_in_sample ( start=0, end=(len ( ts_df ) - 1) ), index=ts_df.index )
        """
        Residuals
        """        
        if self.model['method'] in ['lin','auto_arima','ses','es']:
            try:
                # use fittedvalues to fill in the model dictionary
                self.model['residuals'] = pd.Series (
                    np.asarray ( ts_df.y ) - np.asarray ( self.model['fittedvalues'] ).flatten(), index=ts_df.index )
                self.model['upper_whisker_res'] = self.model['residuals'].mean () + 1.5 * (
                        self.model['residuals'].quantile ( 0.75 ) - self.model['residuals'].quantile ( 0.25 ))
            except (KeyError, AttributeError) as e:
                print ( "Model was not fitted or ts has other structure....: {}".format ( e ) )
            #            
            self.model['lower_conf_int'] = None
            self.model['upper_conf_int'] = None
        elif self.model['method'] == 'prophet':
            try:
                # use fittedvalues to fill in the model dictionary
                self.model['residuals'] = pd.Series (
                    np.asarray ( ts_df.y ) - np.asarray ( self.model['fittedvalues']['yhat'] ), index=self.model['train_dt'].index )
                self.model['upper_whisker_res'] = None
            except (KeyError, AttributeError) as e:
                print ( "Model was not fitted or ts has other structure....: {}".format ( e ) )
            #   
            self.model['lower_conf_int'] = pd.Series ( np.asarray ( self.model['fittedvalues']['yhat_lower'] ), index = self.model['train_dt'].index)
            self.model['upper_conf_int'] = pd.Series ( np.asarray ( self.model['fittedvalues']['yhat_upper'] ), index = self.model['train_dt'].index)
        #       
        self.model['forecast'] = None
        self.model['residuals_forecast'] = None

    def ts_diagnose(self):
        """
        
        Model diagnostic plots
        """
        try:
            assert not self.model['m_fit'] is None
        except AssertionError:
            print ( "Model has to be fitted first! Please call ts_fit(...)" )
            sys.exit ( "STOP" )

        if self.model['method'] == 'auto_arima':
            self.model['m_fit'].plot_diagnostics ( figsize=(9, 3.5) )
        
        self.plot_residuals ()
        
        if self.model['method'] == 'prophet':
            if self.model_params['diagnose']:
                if input ( "Run cross validation y/n? Note, depending on parameters provided this can take some time..." ).strip().lower() == 'y':
                    start = time()
                    print("Running cross validation using parameters provided....")
                    if 'history' in list ( self.model_params.keys () ) and not self.model_params['history'] is None and self.model_params[
                        'history'] != '':
                        try:
                            self.model['prophet_cv'] = cross_validation ( self.model['m_fit'], initial=self.model_params['history'],
                                                                          period=self.model_params['period'],
                                                                          horizon=self.model_params['horizon'] )
                        except:
                            print ( "Prophet cross validation error: check your parameters 'history', 'horizon', 'period'!" )
                            sys.exit ( "STOP" )
                    else:
                        try:
                            self.model['prophet_cv'] = cross_validation ( self.model['m_fit'], period=self.model_params['period'],
                                                                          horizon=self.model_params['horizon'] )
                        except:
                            print ( "Prophet cross validation error: check your parameters 'horizon', 'period'!" )
                            sys.exit ( "STOP" )
                            
                    print( "Time elapsed: {}".format ( time()-start ) )
                    simu_intervals = self.model['prophet_cv'].groupby('cutoff')['ds'].agg (
                        [('forecast_start', 'min'),
                         ('forecast_till', 'max')] )
                    print("Following time windows and cutoffs have been set-up:\n")
                    print(simu_intervals)
                    #
                    plot_cross_validation_metric(self.model['prophet_cv'], metric='mape')
                    #
                    print("Running performance metrics...")
                    self.model['prophet_p'] = performance_metrics(self.model['prophet_cv']) 
                    #
                    """
                    perf_met = np.asarray(self.model['prophet_p']['horizon'].map ( str ). apply(lambda x: float(x.split(' ')[0])))
                    fig, ax = plt.subplots()
                    plt.plot(perf_met, self.model['prophet_p']['mape'])
                    fig.canvas.draw()
                    #change ticks
                    labels = [item.get_text() for item in ax.get_xticklabels()]
                    unit = ' days'
                    if self._freq == 'S':
                        unit = ' secs'
                    elif self._freq == 'min':
                        unit = ' mins'
                    elif self._freq == 'H':
                        unit = ' hours'
                    elif self._freq == 'W':
                        unit = ' weeks'
                    elif self._freq == 'M':
                        unit = ' months'
                    ax.set_xticklabels(map(lambda x: x + unit, labels))
                    
                    plt.ylabel('MAPE')
                    plt.title('MAPE over different forecasting periods.')
                    plt.xticks(rotation=45)
                    plt.show()
                    """
                    
                else:     
                    print("OK")
                    return

    def plot_residuals(self):
        """
        
        Plot the resuls
        """
        try:
            assert not self.model['m_fit'] is None
        except AssertionError:
            print ( "Model has to be fitted first! Please call ts_fit(...)" )
            sys.exit ( "STOP" )

        fig, axes = plt.subplots ( 2, 1, figsize=(20, 5), sharex=True )
        # 
        if self.model['method'] == 'prophet':
            axes[0].plot ( pd.Series ( np.asarray ( self.model['fittedvalues']['yhat'] ), index=self.model['train_dt'].index ) , color='y', linewidth=2.0 ) 
            axes[0].plot ( pd.Series ( np.asarray ( self.model['train_dt']['y'] ), index=self.model['train_dt'].index ), color='b' )
        else:                 
            axes[0].plot ( self.model['fittedvalues'], color='y' )
            axes[0].plot ( self.model['train_dt'], color='b' )
            
        axes[0].set_ylabel ( "Model Fit" )
        axes[0].set_title ( "Real (blue) and estimated values" )
        #
        axes[1].plot ( self.model['residuals'], color="r" )
        if not self.model['lower_conf_int'] is None and not self.model['upper_conf_int'] is None:
            axes[0].fill_between ( self.model['lower_conf_int'].index,
                                   self.model['lower_conf_int'],
                                   self.model['upper_conf_int'],
                                   color='k', alpha=.15 )
        if not self.model['upper_whisker_res'] is None:
            axes[1].axhline ( y=self.model['upper_whisker_res'],
                              xmin=0,
                              xmax=1, color='m', label='upper_whisker',
                              linestyle='--', linewidth=1.5 )
            axes[1].axhline ( y=-self.model['upper_whisker_res'],
                              xmin=0,
                              xmax=1, color='m', label='upper_whisker',
                              linestyle='--', linewidth=1.5 )

        axes[1].set_ylabel ( 'Residuals' )
        axes[1].set_title (
           'Difference between model output and the real data and +/- upper whisker' )
        
        plt.gcf ().autofmt_xdate ()
        plt.grid ( True )
        plt.show ()
            
    def ts_test(self):
        """
        
        Validate the fitted model if test data available
        """
        try:
            assert not self.model['m_fit'] is None
        except AssertionError:
            print ( "Model has to be fitted first! Please call ts_fit(...)" )
            sys.exit ( "STOP" )

        try:
            assert not self.model['test_dt'] is None
        except (KeyError, AssertionError):
            print ( "Nothing to validate. Call ts_forecast() or specify amount of training data when initializing the object." )
            return
        else:
            ts_test_df = self.model['test_dt']
            n_forecast = len ( ts_test_df )

            if self.model['method'] =='lin':
                x_future = np.arange ( len ( self.model['train_dt'] ), len ( self.model['train_dt'] ) + len ( ts_test_df ) ).reshape ( -1,1 )
                self.model['forecast'] = pd.Series ( self.model['m_fit'].predict(x_future), index=ts_test_df.index )
                self.model['lower_conf_int'] = None
                self.model['upper_conf_int'] = None
            elif self.model['method'] == 'auto_arima':
                future, confint = self.model['m_fit'].predict ( n_periods=n_forecast, return_conf_int=True )
                self.model['forecast'] = pd.Series ( future, index=ts_test_df.index )
                self.model['lower_conf_int'] = pd.Series ( confint[:, 0], index=ts_test_df.index )
                self.model['upper_conf_int'] = pd.Series ( confint[:, 1], index=ts_test_df.index )
            elif self.model['method'] in ['ses', 'es']:
                self.model['forecast'] = self.model['m_fit'].forecast ( n_forecast )
                self.model['lower_conf_int'] = None
                self.model['upper_conf_int'] = None
            elif self.model['method'] == 'prophet':
                self.model['forecast'] = self.model['m'].predict ( ts_test_df.copy().reset_index ().drop ('y', axis=1)  )
                self.model['lower_conf_int'] = pd.Series ( np.asarray ( self.model['forecast']['yhat_lower'] ), index = ts_test_df.index)
                self.model['upper_conf_int'] = pd.Series ( np.asarray ( self.model['forecast']['yhat_upper'] ), index = ts_test_df.index)
            #  
            if self.model['method'] in ['lin','auto_arima','ses','es']:
                self.model['residuals_forecast'] = pd.Series (
                    np.asarray ( ts_test_df.y ) - np.asarray ( self.model['forecast'] ),
                    index=ts_test_df.index )
            elif self.model['method'] == 'prophet':
                self.model['residuals_forecast'] = pd.Series (
                    np.asarray ( ts_test_df.y ) - np.asarray ( self.model['forecast']['yhat'] ),
                    index=ts_test_df.index )
            self.plot_forecast ()
            
    def ts_forecast(self, n_forecast):
        """
        
        Forecast time series over time frame in the future specified via n_forecast
        """
        #
        try:
            n_forecast = int ( n_forecast )
            assert 0 < n_forecast < len ( self.ts_df )
        except AssertionError:
            print ( "Number of periods to be forecasted is too low, too high or not numeric!" )
            sys.exit ( "STOP" )
        except ValueError:
            print ( "n_forecast must be convertable to int type!" )
            sys.exit ( "STOP" )
        else:
            n_forecast = int ( n_forecast )

        if self._p_train == 1:    
            try:
                assert not self.model['m_fit'] is None
            except AssertionError:
                print ( "Model has to be fitted first! Please call ts_fit(...)" )
                sys.exit ( "STOP" )
        #        
        print("Fitting using all data....")
        self.ts_fit(method=self.model['method'], 
                    dict_params=self.model_params,
                    mode='forecast')
        

        print("Forecasting next "+str(n_forecast)+str(self._freq))    
        if self.model['method'] in ['auto_arima', 'lin']:
            if self.model['method'] == 'auto_arima':
                future, confint = self.model['m_fit'].predict ( n_periods=n_forecast, return_conf_int=True )
            else:
                x_future = np.arange ( len ( self.model['train_dt'] ), len ( self.model['train_dt'] ) + n_forecast ).reshape ( -1,1 )
                future = self.model['m_fit'].predict(x_future)
            #
            if self._freq == 'S':
                idx_future = pd.date_range ( start=max ( self.model['train_dt'].index ),
                                             end=max ( self.model['train_dt'].index ) + datetime.timedelta (
                                                 seconds=n_forecast - 1 ), freq='S' )
            elif self._freq == 'min':
                idx_future = pd.date_range ( start=max ( self.model['train_dt'].index ),
                                             end=max ( self.model['train_dt'].index ) + datetime.timedelta (
                                                 minutes=n_forecast - 1 ), freq='min' )
            elif self._freq == 'H':
                idx_future = pd.date_range ( start=max ( self.model['train_dt'].index ),
                                             end=max ( self.model['train_dt'].index ) + datetime.timedelta (
                                                 hours=n_forecast - 1 ), freq='H' )
            elif self._freq == 'D':
                idx_future = pd.date_range ( start=max ( self.model['train_dt'].index ),
                                             end=max ( self.model['train_dt'].index ) + datetime.timedelta (
                                                 days=n_forecast - 1 ), freq='D' )
            elif self._freq == 'W':
                idx_future = pd.date_range ( start=max ( self.model['train_dt'].index ),
                                             end=max ( self.model['train_dt'].index ) + datetime.timedelta (
                                                 weeks=n_forecast - 1 ), freq='W' )
            elif self._freq == 'M':
                idx_future = pd.date_range ( start=max ( self.model['train_dt'].index ),
                                             end=max ( self.model['train_dt'].index ) + relativedelta ( months=+(n_forecast - 1) ),
                                             freq='M' )
            # fill in model
            self.model['forecast'] = pd.Series ( future, index=idx_future )
            if self.model['method'] == 'auto_arima':
                self.model['lower_conf_int'] = pd.Series ( confint[:, 0], index=idx_future )
                self.model['upper_conf_int'] = pd.Series ( confint[:, 1], index=idx_future )
            else:
                self.model['lower_conf_int'] = None
                self.model['upper_conf_int'] = None
        elif self.model['method'] in ['ses', 'es']:
            self.model['forecast'] = self.model['m_fit'].forecast ( n_forecast )
            self.model['lower_conf_int'] = None
            self.model['upper_conf_int'] = None
        elif self.model['method'] == 'prophet':
            future = self.model['m'].make_future_dataframe ( periods=n_forecast, freq=self._freq ) 
            if self.model_params['weekend_seasonality']:
                future['on_weekend'] = future['ds'].apply ( we_season )
                future['off_weekend'] = ~future['ds'].apply ( we_season ) 
            
            self.model['forecast'] = self.model['m'].predict ( future )
            self.model['lower_conf_int'] = pd.Series ( np.asarray ( self.model['forecast']['yhat_lower'] ), index = future.ds)
            self.model['upper_conf_int'] = pd.Series ( np.asarray ( self.model['forecast']['yhat_upper'] ), index = future.ds)

        self.model['residuals_forecast'] = None
        self.plot_forecast ()
        
    def plot_forecast(self):
        """
        
        Plot forecasted values
        """
        try:
            assert not self.model['m_fit'] is None
        except AssertionError:
            print ( "Model has to be fitted first! Please call ts_fit(...)" )
            sys.exit ( "STOP" )
        #     
        try:
            assert not self.model['forecast'] is None
        except AssertionError:
            print ( "Neither ts_validation(...) nor ts_forecast(...) have been called yet!" )
            sys.exit ( "STOP" )

        if (self.model['method'] in ['lin', 'auto_arima', 'ses', 'es']) or (
                self.model['method'] == 'prophet' and not self.model['residuals_forecast'] is None):
            
            fig, axes = plt.subplots ( 2, 1, figsize=(20, 7), sharex=True )
            # 
            if self.model['method'] == 'prophet':
                axes[0].plot ( pd.Series ( np.asarray ( self.model['fittedvalues']['yhat'] ), index=self.model['train_dt'].index ) , color='y', linewidth=2.0 ) 
                axes[0].plot ( pd.Series ( np.asarray ( self.model['train_dt']['y'] ), index=self.model['train_dt'].index ), color='b' )
            else:                 
                axes[0].plot ( self.model['fittedvalues'], color='y' )
                axes[0].plot ( self.model['train_dt'], color='b' )    
            #
            if not self.model['residuals_forecast'] is None:
                axes[0].plot ( self.ts_df, color='b' )
            if self.model['method'] == 'prophet':
                axes[0].plot ( pd.Series ( np.asarray ( self.model['forecast']['yhat'] ), index=self.model['test_dt'].index ) , color='darkgreen' )    
            else:                                                      
                axes[0].plot ( self.model['forecast'], color='darkgreen' )
                
            if not self.model['lower_conf_int'] is None and not self.model['upper_conf_int'] is None:
                axes[0].fill_between ( self.model['lower_conf_int'].index,
                                       self.model['lower_conf_int'],
                                       self.model['upper_conf_int'],
                                       color='k', alpha=.15 )
            axes[0].set_ylabel ( "Fit and Forecast/Validation" )
            axes[0].set_title ( "Real (blue), estimated (yellow) and forecasted values" )
            #
            if not self.model['residuals_forecast'] is None:
                axes[1].plot ( pd.concat ( [self.model['residuals'], self.model['residuals_forecast']], axis=0 ),
                               color='r' )
            axes[1].plot ( self.model['residuals'], color="r" )
            
            if not self.model['upper_whisker_res'] is None:
                axes[1].axhline ( y=self.model['upper_whisker_res'],
                                  xmin=0,
                                  xmax=1, color='m',
                                  label='upper_whisker',
                                  linestyle='--', linewidth=1.5 )
                axes[1].axhline ( y=-self.model['upper_whisker_res'],
                                  xmin=0,
                                  xmax=1, color='m',
                                  label='upper_whisker',
                                  linestyle='--', linewidth=1.5 )
            axes[1].set_ylabel ( 'Residuals' )
            axes[1].set_title (
                'Difference between model output and the real data both, for fitted and forecasted and +/- upper whisker or confidence intervals')
            plt.gcf ().autofmt_xdate ()
            plt.grid ( True )
            plt.show ()
        if self.model['method'] == 'prophet': #and self.model['residuals_forecast'] is None:
            fig_forecast = self.model['m'].plot(self.model['forecast'])
            fig_components = self.model['m'].plot_components(self.model['forecast'])
            if self.model_params['add_change_points']:
                a = add_changepoints_to_plot(fig_forecast.gca(), self.model['m'], self.model['forecast'])

    plt.gcf ().autofmt_xdate ()
    plt.grid ( True )
    plt.show ()    
                
    def ts_decompose(self, params=None):
        """
        
        Decomposes time series into trend, seasonal 
        """
        if params is None:
            params = dict({'model': 'additive',
                          'freq':1})
        try:
            assert isinstance ( params, dict ) 
        except AssertionError:
            print("Dictionary is expected for parameters!")
            sys.exit( "STOP" )
        
        try:
            assert 'model' in list ( params.keys() )
        except AssertionError:
            print("Unexpected dictionary keys. At least decomposition model must be supplied!")
            sys.exit( "STOP" )
         
        if not 'freq' in list ( params.keys() ):
            params['freq'] = 1
         
        try:    
            if not self.ts_df.index.freq is None:
                res = seasonal_decompose ( self.ts_df.loc[:, 'y'], model=params['model'] )     
            else:
                res = seasonal_decompose ( self.ts_df.loc[:, 'y'], model=params['model'], freq=params['freq'] )     
                
        except ValueError as e:
            print ( "seasonal_decompose error... {}". format ( e ) )
        else: 
            self.model['seasonal'] = res.seasonal
            self.model['trend'] = res.trend
            self.model['baseline'] = self.model['seasonal'] + self.model['trend']
            self.model['residuals'] = res.resid
            self.model['upper_whisker_res'] = self.model['residuals'].mean () + 1.5 * (
                    self.model['residuals'].quantile ( 0.75 ) - self.model['residuals'].quantile ( 0.25 ))
        
    def plot_decompose(self):
        try:
            assert 'seasonal' in list ( self.model.keys() )
        except AssertionError:
            #print("Decomposition not yet performed! Calling ts_decompose() first...")
            self.ts_decompose()
       
        fig, axes = plt.subplots ( 4, 1, figsize=(20, 7), sharex=True )
        axes[0].plot(self.model['trend'])
        axes[0].set_title("Trend")
        #
        axes[1].plot(self.model['seasonal'])
        axes[1].set_title("Seasonality")
        #
        axes[2].plot(self.model['baseline'])
        axes[2].set_title("Baseline")
        #
        axes[3].plot(self.model['residuals'])
        axes[3].set_title("Residuals")
        #
        if not self.model['upper_whisker_res'] is None:
            axes[3].axhline ( y=self.model['upper_whisker_res'],
                              xmin=0,
                              xmax=1, color='m',
                              label='upper_whisker',
                              linestyle='--', linewidth=1.5 )
            axes[3].axhline ( y=-self.model['upper_whisker_res'],
                              xmin=0,
                              xmax=1, color='m',
                              label='upper_whisker',
                              linestyle='--', linewidth=1.5 )
        
           
        plt.gcf ().autofmt_xdate ()
        plt.grid ( True )
        plt.show ()    

#EoF class
def print_attributes(obj):
    """
    
    Prints attributes ob catsed object obj
    """
    for attr in obj.__dict__:
        print ( attr, getattr ( obj, attr ) )

def get_input_args():
    """
    Retrieves and parses the 6 command line arguments provided by the user when
    they run the program from a terminal window. 
    """

    parser = argparse.ArgumentParser ()
    # 
    parser.add_argument ( '--dt_path', type=str, default='ts.csv', help="path to your time series data" )
    parser.add_argument ( '--value_column', type=str, default='Value', help="Value column name" )
    parser.add_argument ( '--time_format', type=str, default='%Y-%m-%d %H:%M:%S', help="time format" )
    parser.add_argument ( '--freq', type=str, default='H', help="time series modelling frequency" )
    parser.add_argument ( '--transform', type=str, default='', help="time series transformation" )
    parser.add_argument ( '--p_train', type=float, default=0.7, help="size of test data" )
    parser.add_argument ( '--method', type=str, default='auto_arima', help="method to use" )
    parser.add_argument ( '--n_forecast', type=int, default=5, help="number of periods to be forecasted" )

    return parser.parse_args ()


def main():
    args = get_input_args ()
    
    try:
        assert args.dt_path.split ( '.' )[-1] == 'csv'
    except AssertionError:
        print ( "Here only csv supported!" )
        sys.exit ( "STOP" )
    try:
        #ts = pd.read_csv ( args.dt_path, delimiter=';', index_col=0, decimal=',' )
        ts_df = pd.read_csv( args.dt_path, index_col='Date', delimiter=',', usecols=['Date',args.value_column], parse_dates=True)        
        print("Data of shape "+str(ts_df.shape) + " read in.")
    except IOError:
        print ( "File could not be read!" )
        sys.exit ( "STOP" )
    except:
        print ( "Incompatible file format! Expected columns 'Date' and 'Value'" )
        sys.exit ( "STOP" ) 

    # initiate
    uv_tsf = UVariateTimeSeriesForecaster (
        dict ( {'ts_df': ts_df, 'time_format': args.time_format, 'freq': args.freq, 'transform': args.transform, 'p_train': args.p_train} ) )
    print_attributes ( uv_tsf )
    # resample
    uv_tsf.ts_resample ()
    if args.method == 'prophet':
        prophet_params = dict({"diagnose": True})
        if input("Would you like to diagnose prophet (cross validation) y/n?").strip().lower() == 'y':
            history = input("Enter history of format, e.g., '725 days' or '10000 hours'. Empty value would mean, 3X horizon will be considered: ".strip().lower())
            if history == '':
                prophet_params['history'] = None
            else:
                prophet_params['history'] = history
            #
            horizon = input("Enter horizon of format, e.g., '365 days' or '10 hours': ".strip().lower())
            try:
                assert not horizon == ''
            except:
                print("This value cannot be empty!")
                sys,exit ( "STOP" )
            else:
                prophet_params['horizon'] = horizon
            #    
            period = input("Enter period of format, e.g., '180 days' or '5 hours': ".strip().lower())
            try:
                assert not period == ''
            except:
                print("This value cannot be empty!")
                sys,exit ( "STOP" )
            else:
                prophet_params['period'] = horizon
     
        uv_tsf.set_model_params(prophet_params)
        
    if input ( "Continue with ts_fit y/n?" ).strip().lower() == 'y':                   
        # fit
        uv_tsf.ts_fit ( method=args.method )
    else:
        sys.exit( "OK" )
    if input ( "Continue with ts_diagnose y/n?" ).strip().lower() == 'y':                   
        # diagnose
        uv_tsf.ts_diagnose ()
    else:
        sys.exit( "OK" )
    if input ( "Continue with ts_test y/n?" ).strip().lower() == 'y':                   
        # test
        uv_tsf.ts_test ()
    else:
        sys.exit( "OK" )
    if input ( "Continue with ts_forecast y/n?" ).strip().lower() == 'y':                   
        # forecast
        uv_tsf.ts_forecast ( n_forecast=args.n_forecast )
    else:
        sys.exit( "OK" )


if __name__ == '__main__':
    # call main
    main ( )
