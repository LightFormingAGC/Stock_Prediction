
import warnings
import os
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from inspect import trace
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import allSecTickGen
from tqdm.auto import tqdm
import datetime
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()

warnings.filterwarnings("ignore")


def statsGen(data):

    adjClose = data.iloc[:, :2]
    result = coint_johansen(adjClose, det_order=0, k_ar_diff=1)

    return result.lr1, result.cvt, result.lr2, result.cvm


def generator(date):

    # check if there is a csv file under folder Mean_Reverting
    # if os.path.exists('cointegrated.csv'):
    #     data = pd.read_csv('cointegrated.csv')
    #     print('cointegrated.csv loaded')
    # else:
    #     data = pd.DataFrame(columns=[
    #                         'pair', 'cointegrated times', 'cointegrated checks',
    #                         'cointegrated ratio', 'last check date', 'one-p-z-stats'
    #     ])
    cointegrated_pairs = []

    df = allSecTickGen.get_tickers()

    for row in tqdm(df.iterrows(), total=len(df)):

        try:
            pair = (row[1]['ticker1'], row[1]['ticker2'])
            # end date is start date
            prices = pdr.get_data_yahoo(pair, end=date)[
                ['Adj Close', 'Volume']]

        except:
            continue

        prices = prices.dropna()

        cond1 = len(prices) < 2000
        if cond1:
            continue

        cond2 = prices['Adj Close'].iloc[-1,
                                         0] < 5 or prices['Adj Close'].iloc[-1, 1] < 5
        cond3 = prices['Volume'].iloc[-1,
                                      0] < 40000000 or prices['Volume'].iloc[-1, 1] < 4000000
        if cond2 or cond3:
            continue

        info = statsGen(prices['Adj Close'])
        lr_stats = info[0][0]
        lr_95pct_threshold = info[1][0][1]
        mle_stats = info[2][0]
        mle_95pct_threshold = info[3][0][1]
        if lr_stats > lr_95pct_threshold or mle_stats > mle_95pct_threshold:
            cointegrated_pairs.append(pair)
        # if pair in data['pair'].values:
        #     print('pair already in data')
        #     data.loc[data['pair'] == pair, 'cointegrated checks'] += 1
        #     info= statsGen(prices.loc[:start_date])
        #     lr_stats= info[0][0]
        #     lr_90pct_threshold= info[1][0][0]
        #     mle_stats= info[2][0]
        #     mle_90pct_threshold= info[3][0][0]
        #     if lr_stats > lr_90pct_threshold or mle_stats > mle_90pct_threshold:
        #         data.loc[data['pair'] == pair,
        #                  'cointegrated times'] += 1
        #     data.loc[data['pair'] == pair, 'cointegrated ratio'] = data.loc[data['pair'] == pair, 'cointegrated times'] /
        #         data.loc[data['pair'] == pair, 'cointegrated checks']
        #     data.loc[data['pair'] == pair, 'last check date']= start_date
        #     data.loc[data['pair'] == pair, 'one-p-z-stats']= one_p_z_test(
        #         data.loc[data['pair'] == pair, 'cointegrated ratio'], 0.8, data.loc[data['pair'] == pair, 'cointegrated checks'])

        # else:
        #     print('pair not in data', pair)
        #     print(data['pair'].values)
        #     # stationary test 90% confidence
        #     prices['stationary']= 0
        #     for j in range(50, len(prices)):

        #         try:
        #             info= statsGen(prices.iloc[:j, :])
        #             lr_stats= info[0][0]
        #             lr_90pct_threshold= info[1][0][0]
        #             mle_stats= info[2][0]
        #             mle_90pct_threshold= info[3][0][0]
        #             if lr_stats > lr_90pct_threshold or mle_stats > mle_90pct_threshold:
        #                 prices.iloc[j, -1]= 1
        #         except:
        #             continue

            # cointegrated_times= prices['stationary'].sum()
            # conintegrated_checks= len(prices) - 50
            # cointegrated_ratio= cointegrated_times / conintegrated_checks
            # data= data.append({'pair': pair, 'cointegrated times': cointegrated_times,
            #                     'cointegrated checks': conintegrated_checks, 'cointegrated ratio': cointegrated_ratio,
            #                     'last check date': start_date, 'one-p-z-stats': one_p_z_test(cointegrated_ratio, 0.8, conintegrated_checks)},
            #                    ignore_index=True)

    return cointegrated_pairs


# generator('2021-05-15')
