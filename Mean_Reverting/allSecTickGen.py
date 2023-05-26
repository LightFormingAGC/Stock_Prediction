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
        'Enter a backtest date in the format of YYYY-MM-DD: \n otherwise empty is yesterday')

    # if dates == '':
    # previous trading day
    if dates == '':
        dates = (pd.Timestamp.today() - pd.Timedelta(days=1)
                 ).strftime('%Y-%m-%d')  # yesterday
    # headers = {
    #     'authority': 'api.nasdaq.com',
    #     'accept': 'application/json, text/plain, */*',
    #     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
    #     'origin': 'https://www.nasdaq.com',
    #     'sec-fetch-site': 'same-site',
    #     'sec-fetch-mode': 'cors',
    #     'sec-fetch-dest': 'empty',
    #     'referer': 'https://www.nasdaq.com/',
    #     'accept-language': 'en-US,en;q=0.9',
    # }

    # params = (
    #     ('tableonly', 'true'),
    #     ('limit', '25'),
    #     ('offset', '0'),
    #     ('download', 'true'),
    # )

    # r = requests.get('https://api.nasdaq.com/api/screener/stocks',
    #                  headers=headers, params=params)

    # data = r.json()['data']
    # df = pd.DataFrame(data['rows'], columns=data['headers'])
    # df = df[df['marketCap'].str.len() > 5]
    # df['marketCap'] = df['marketCap'].astype(float)
    # df = df[df['marketCap'] > 1000000000]
    # df.volume = df.volume.astype(int)
    # df = df[df.volume > 4000000]

    # # filter stocks with last sale price > 5
    # # get rid the $ sign
    # df['lastsale'] = df['lastsale'].str.replace('$', '', regex=False)
    # df['lastsale'] = df['lastsale'].astype(float)
    # df = df[df['lastsale'] > 5]

    # # group by sector and aggregate all symbol into a list
    # df = df.groupby('sector').agg({'symbol': lambda x: list(x)})
    # df = df[df.symbol.str.len() > 5]
    # # for each sector, geenrator all possible pairs using itertools.combinations
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

            if len(data) < 1500:
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
    df.to_csv('cointegrated_pairs.csv', index=False)
    with open('datas.pkl', 'wb') as f:
        pickle.dump(data, f)
    # results = pd.concat(results)
    # results.to_csv('cointegrated_pairs.csv', index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
# check volumn
# pass的store price volumns with dates, so when backtest we can use the price and volumns with dates directly without re-download
