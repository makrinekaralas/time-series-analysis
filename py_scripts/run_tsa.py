#!/usr/bin/env python.
__author__ = "Maka Karalashvili"
__copyright__ = "BMW Group"
__version__ = "0.0.1"
__maintainer__ = "Maka Karalashvili"
__email__ = "maka.karalashvili@bmw.de"
__status__ = "Development"

from tsa import Logger

from tsa import AutoARIMAForecaster

import sys
import pandas as pd
import argparse


def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the auto_arima fit from a terminal window.
    """

    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--dt_path', type=str, default='ts.csv', help="path to your time series data")
    parser.add_argument('--value_column', type=str, default='Value', help="Value column name")
    parser.add_argument('--time_format', type=str, default='%Y-%m-%d %H:%M:%S', help="time format")
    parser.add_argument('--freq', type=str, default='H', help="time series modelling frequency")
    parser.add_argument('--transform', type=str, default='', help="time series transformation")
    parser.add_argument('--n_test', type=int, default=0, help="size of test data in  units of 'freq'")
    parser.add_argument('--n_val', type=int, default=0, help="size of test data in units of 'freq'")
    parser.add_argument('--n_forecast', type=int, default=5, help="number of periods to be forecasted")
    #
    parser.add_argument('--start_p', type=int, default=1, help="start p for auto_arima")
    parser.add_argument('--max_p', type=int, default=3, help="max p for auto_arima")
    parser.add_argument('--d', type=int, default=0, help="d for auto_arima")
    parser.add_argument('--start_q', type=int, default=1, help="start q for auto_arima")
    parser.add_argument('--max_q', type=int, default=3, help="max q for auto_arima")
    #
    parser.add_argument('--start_P', type=int, default=1, help="start P for auto_arima")
    parser.add_argument('--max_P', type=int, default=1, help="max P for auto_arima")
    parser.add_argument('--D', type=int, default=0, help="D for auto_arima")
    parser.add_argument('--start_Q', type=int, default=1, help="start Q for auto_arima")
    parser.add_argument('--max_Q', type=int, default=1, help="max Q for auto_arima")
    parser.add_argument('--seasonal', type=bool, default=False, help="seasonal component for auto_arima")

    parser.add_argument('--suppress', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, nargs='?',
                        help="suppress outputs")

    return parser.parse_args()


def main():
    args = get_input_args()
    main_logger = Logger("auto_arima app")

    try:
        assert args.dt_path.split('.')[-1] == 'csv'
    except AssertionError:
        main_logger.exception("Here only csv supported!")
        sys.exit("STOP")
    try:
        ts_df = pd.read_csv(args.dt_path, index_col='Date', delimiter=',', usecols=['Date', args.value_column],
                            parse_dates=True)
        main_logger.info("Data of shape {0} read in.".format(str(ts_df.shape)))
    except IOError:
        main_logger.exception("File could not be read!")
        sys.exit("STOP")
    except (NameError, KeyError):
        main_logger.exception("Incompatible file format! Expected columns 'Date' and "+ args.value_column)
        sys.exit("STOP")

    # initiate
    tsf_obj = AutoARIMAForecaster(ts_df=ts_df,
                                  time_format=args.time_format,
                                  freq=args.freq,
                                  n_test=args.n_test,
                                  n_val=args.n_val)
    if args.transform != '':
        tsf_obj.ts_transform(args.transform)

    if input("Continue with ts_fit y/n?").strip().lower() == 'y':
        tsf_obj.ts_fit(suppress=args.suppress)
    else:
        main_logger.info("OK")

    if input("Continue with ts_diagnose y/n?").strip().lower() == 'y':
        tsf_obj.ts_diagnose()
    else:
        main_logger.info("OK")
    if input("Continue with ts_test y/n?").strip().lower() == 'y':
        tsf_obj.ts_test()
    else:
        main_logger.info("OK")
    if input("Continue with ts_forecast y/n?").strip().lower() == 'y':
        tsf_obj.ts_forecast(n_forecast=args.n_forecast)
    else:
        main_logger.info("OK")


if __name__ == '__main__':
    # call main
    main()


