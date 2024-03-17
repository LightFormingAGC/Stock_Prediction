import multiprocessing
import itertools
import requests
import pandas as pd
from polygon import RESTClient
import pandas as pd
from polygonAPIkey import polygonAPIkey
from pandas_datareader import data as pdr
from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np
import yfinance as yf
import time
import pickle

import warnings
warnings.filterwarnings('ignore')

yf.pdr_override()


def get_tickers():
    dates = input(
        'Enter a backtest date in the format of YYYY-MM-DD, \n otherwise empty is yesterday: ')

    # if dates == '':
    # previous trading day
    if dates == '':
        dates = (pd.Timestamp.today() - pd.Timedelta(days=1)
                 ).strftime('%Y-%m-%d')  # yesterday

    # df = df.reset_index()
    client = RESTClient(polygonAPIkey)
    mktData = pd.DataFrame(client.get_grouped_daily_aggs(
        date=dates, locale='us', market_type='stocks'))

    return mktData.iloc[:, :]


def tickers_generator(mktData):
    tickers_unique = mktData.ticker.unique()
    backtest_tickers = set()
    tickers_info = {}
    for ticker in tqdm(tickers_unique):
        try:
            data = pdr.get_data_yahoo(ticker)[
                ['Adj Close', 'Volume']]
            data = data.resample('M').last()

            if len(data) < 60:
                continue

        except:
            continue

        data['transaction amount'] = data['Adj Close'] * data['Volume']
        if data['Volume'].max() < 4000000 or data['Adj Close'].max() < 5 or \
                data['transaction amount'].max() < 40000000:
            continue
        else:
            backtest_tickers.add(ticker)
            tickers_info[ticker] = data
    df = pd.DataFrame(backtest_tickers, columns=['ticker'])
    return df, tickers_info


if __name__ == '__main__':

    start_time = time.time()

    all_ticks = get_tickers()

    df, data = tickers_generator(all_ticks)
    # results is set, combine them into one set

    # store tickers list
    df.to_csv('tickers.csv', index=False)

    # store adj prices and volumns with dates for each ticker
    with open('prices.pkl', 'wb') as f:
        pickle.dump(data, f)

    # results = pd.concat(results)
    # results.to_csv('cointegrated_pairs.csv', index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
# check volumn
# pass的store price volumns with dates, so when backtest we can use the price and volumns with dates directly without re-download
