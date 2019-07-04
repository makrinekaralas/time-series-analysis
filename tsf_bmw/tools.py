import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from tsf_bmw.uvtsf import UVariateTimeSeriesForecaster

#@ToDo: implement moving_window, window_step, window_step_size
def moving_val_forecast(uvtsf_obj, granularity='p_week', 
                        start_with=2, moving_window=False,
                        window_step = 0, window_step_size=0, 
                        rem_we=True, method='auto_arima'):
    """
    
    Realizes a simple moving fitting and forecasting algorithm. Data of 'granularity' level 
    are used as test and forecasted over the time frame of granularity level.
    The aim of this algorithm is to analyze and visualize possible deviations in univariate time series
    over time. Example, increase of temperature over 11 weeks.
    
    Parameters:
        uvtsf_obj  - object of UVariateTimeSeriesForecaster class
        
        granularity - defines which granularity to use, p_week and p_day possible, to generate data chunks. Means, that number of weeks/days will be used 
        as test data and one granularity - hence 1 week/day - will be forecasted. If real data available over this forecasted 
        time frame, comparison can be made if the forecasted values fit the real ones.  
        
        start_with - number that defines with what amount of test data to start. In each iteration test data is either increased (moving_window = False)
        or moved via window_step and data of window_step_size is used as test data.
        
        moving_window - True uses moving window approach, whereas when False test data volume is each time increased via time frame of granularity. 
        Hence, at  the end of the itertion all data is used as test data.
        
        window_step - used only when moving_window == True. Defines where the test data should start given grenularity. moving_step=2 and granularity == p_week 
        would take slice of the data starting with the second week of given time series and stretch it over window_step_size.
        
        window_step_size - defines amount of data to be used as test data (size of the slice/chunk)
    """
    try:
        assert isinstance(uvtsf_obj, UVariateTimeSeriesForecaster)
    except AssertionError:
        print ( "Object of type UVariateTimeSeriesForecaster is expected!" ) 
        sys.exit ( "STOP" )
    #
    try:
        assert granularity.lower() in ['p_week','p_day']
    except AssertionError:
        print ( "Moving over week or a day is supported!" ) 
        sys.exit ( "STOP" )
        
    #
    ts = uvtsf_obj.ts
    try:
        assert not ts.empty
    except AssertionError:
        print("Provided object exhibits empty time series data!")
        sys.exit ( "STOP" )
    
    p_gran = []
    if granularity.lower() == 'p_week':
        ts['p_gran'] = ts.index.year.map ( str ) + ts.index.week.map ( str )
        ts['p_gran'] = ts['p_gran'].apply( np.int_, errors='coerce' )
    elif granularity.lower() == 'p_day':
        ts['p_gran'] = ts.index.year.map ( str ) + ts.index.month.map ( str ) + ts.index.day.map ( str )
        ts['p_gran'] = ts['p_gran'].apply( np.int_, errors='coerce' )
        
    p_gran = np.unique(ts['p_gran'])
    print(p_gran)
    #
    idx_hist = start_with + 1
    while True:
        cur_uvtsf = copy ( uvtsf_obj )
        hist = p_gran[:idx_hist]
        print(hist)
        print ( "===========================================\n Model: Using historic data of first: " + str (
            idx_hist - 1 ) + " " + granularity.lower () + "s starting from the " + str ( p_gran[0] ) )
        #
        ts_hist = pd.DataFrame ( ts.loc[np.asarray ( tuple ( x in hist for x in ts['p_gran'] ) ), 'y'] )
        ts_hist.sort_index ( inplace=True )
        p_train = 1- len ( ts[ts['p_gran'] == p_gran[idx_hist]] ) / len ( ts_hist )  
        print(p_train)
        #        
        cur_uvtsf._p_train = p_train
        cur_uvtsf.ts = ts_hist
        # print_attributes(cur_uvtsf)
        # fit the model
        cur_uvtsf.ts_fit ( method=method.lower (), rem_we=rem_we )
        cur_uvtsf.diagnose ()
        cur_uvtsf.ts_validate ()
        if input ( "Continue y/n?" ).strip().lower() != 'y':
            sys.exit( "STOP" )
        #
        if (idx_hist) >= (len ( p_gran ) - 1):
            break
        idx_hist += 1
        del cur_uvtsf

        
def moving_linreg(uvtsf_obj, hist='2D', step='1D'):
    """
    
    Moving linear regression over windows of size 'hist' that move with the step 'step'
    Parameters:
        - uvtsf_obj: UVariateTimeSeriesForecaster object
        - hist: amount of historic data over which linear model will be fit. The format of this variable is similar to frequency formats in python
        - step: step size with which the window of size 'hist' will be moving. The format of this variable is similar to frequency formats in python
    """
    try:
        assert isinstance ( uvtsf_obj, UVariateTimeSeriesForecaster )
    except AssertionError:
        print ( "Object of type UVariateTimeSeriesForecaster is expected!" )
        sys.exit ( "STOP" )
    #
    try:
        assert len ( hist ) == 2 and hist.upper ()[0].isdigit () and int ( hist.upper ()[0] ) > 0
    except AssertionError:
        print ( "Wrong format for hist or zero unit provided!" )
        sys.exit ( "STOP" )
    try:
        assert hist.upper ()[1] in ['min', 'H', 'D', 'W']
    except AssertionError:
        print ( "Wrong format for hist provided; Unit in ['H','min','D','W'] expected!" )
        sys.exit ( "STOP" )

    #
    try:
        assert len ( step ) == 2 and step.upper ()[0].isdigit () and int ( step.upper ()[0] ) > 0
    except AssertionError:
        print ( "Wrong format for step or zero unit provided!" )
        sys.exit ( "STOP" )
    try:
        assert step.upper ()[1] in ['min', 'H', 'D', 'W']
    except AssertionError:
        print ( "Wrong format for step provided; Unit in ['H','min','D','W'] expected!" )
        sys.exit ( "STOP" )

        #
    try:
        assert not uvtsf_obj.ts_df.empty
    except AssertionError:
        print ( "Provided object exhibits empty time series data!" )
        sys.exit ( "STOP" )

    if uvtsf_obj.ts_df.index.freq is None:
        print ( "No specific frequency detected. \n" )
        print ( "Time series will be resampled to provided frequency: {}".uvtsf_obj._freq )
        uvtsf_obj.ts_resample ()
    # check for hist
    try:
        assert not (uvtsf_obj.ts_df.index.freq == 'H' and hist.upper ()[1] in ['min']) and not (
                uvtsf_obj.ts_df.index.freq == 'D' and hist.upper ()[1] in ['W'])
    except AssertionError:
        print ( "Unit in hist is not compatible with the time series frequency!" )
        sys.exit ( "STOP" )

    ts_df = uvtsf_obj.ts_df
    #
    ts_start = min ( ts_df.index )
    fittedvalues = pd.DataFrame ()
    slopes = pd.DataFrame ()
    dt_id = 0
    print ( "Starting moving linreg...." )
    while True:
        uv_tsf_cur = copy ( uvtsf_obj )
        #
        if hist.upper ()[1] == 'min':
            ts_end = ts_start + datetime.timedelta ( minutes=int ( hist[0] ) )
        elif hist.upper ()[1] == 'H':
            ts_end = ts_start + datetime.timedelta ( hours=int ( hist[0] ) )
        elif hist.upper ()[1] == 'D':
            ts_end = ts_start + datetime.timedelta ( days=int ( hist[0] ) )
        elif hist.upper ()[1] == 'W':
            ts_end = ts_start + datetime.timedelta ( weeks=int ( hist[0] ) )
        elif hist.upper ()[1] == 'M':
            ts_end = ts_start + relativedelta ( months=+int ( hist[0] ) )
            #
        if ts_end > max ( ts_df.index ):
            ts_end = max ( ts_df.index )

        interval = pd.date_range ( start=ts_start, end=ts_end, freq=uvtsf_obj._freq, closed='left' )
        #
        uv_tsf_cur.ts_df = ts_df.loc[np.asarray ( tuple ( x in interval for x in ts_df.index ) ),]
        uv_tsf_cur.p_train = 1.0
        uv_tsf_cur.ts_fit ( method='lin', suppress=True )
        #
        fittedvalues = fittedvalues.append ( pd.DataFrame ( uv_tsf_cur.model['fittedvalues'] ).assign ( dt_id=dt_id ) )
        slopes = slopes.append (
            pd.DataFrame ( np.repeat ( uv_tsf_cur.model['slope'], len ( interval ) ), index=interval ).assign (
                dt_id=dt_id ) )
        #
        if step.upper ()[1] == 'min':
            ts_start = ts_start + datetime.timedelta ( minutes=int ( step[0] ) )
        elif step.upper ()[1] == 'H':
            ts_start = ts_start + datetime.timedelta ( hours=int ( step[0] ) )
        elif step.upper ()[1] == 'D':
            ts_start = ts_start + datetime.timedelta ( days=int ( step[0] ) )
        elif step.upper ()[1] == 'W':
            ts_start = ts_start + datetime.timedelta ( weeks=int ( step[0] ) )
        elif step.upper ()[1] == 'M':
            ts_start = ts_start + relativedelta ( months=+int ( step[0] ) )
            
        dt_id += 1
        # break
        if ts_start > max ( ts_df.index ):
            break

    print ( "moving linreg done!" )

    # plot
    fittedvalues.columns = ['value', 'dt_id']
    slopes.columns = ['value', 'dt_id']

    fig, axes = plt.subplots ( 2, 1, figsize=(20, 7), sharex=True )
    #
    axes[0].plot ( uvtsf_obj.ts_df.index, uvtsf_obj.ts_df.y )
    #
    groups = fittedvalues.groupby ( ['dt_id'] )
    for a, group in groups:
        sub_group = pd.DataFrame ( group[fittedvalues.columns] )
        axes[0].plot ( sub_group.index, sub_group.value, linewidth=1.3 )
    #    
    slopes_agg = slopes.reset_index ().groupby ( by=['dt_id'] ).agg ( {
        'value': 'mean',
        'index': 'min'
    } )
    slopes_agg.set_index ( 'index', inplace=True )
    axes[1].plot ( slopes_agg.index, slopes_agg.value, marker='o', linestyle='dashed', color = 'r', markersize=10 )
    axes[1].set_title ( "Slopes" )

    plt.gcf ().autofmt_xdate ()
    plt.grid ( True )
    plt.show ()

    return (fittedvalues, slopes)