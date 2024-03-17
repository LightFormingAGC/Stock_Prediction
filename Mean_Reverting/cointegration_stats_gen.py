
import warnings
import os
from tqdm.auto import tqdm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import math
from pandas_datareader import data as wb
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import itertools
from pandas_market_calendars.exchange_calendar_nyse import NYSEExchangeCalendar


def statsGen(data):

    adjClose = data.iloc[:, :2]
    result = coint_johansen(adjClose, det_order=0, k_ar_diff=1)

    return result
    return result.lr1, result.cvt, result.lr2, result.cvm, result.evec


pairs = pd.read_csv('tickers.csv', index_col=False)

with open('prices.pkl', 'rb') as f:
    data = pickle.load(f)


coint_pairs = list(itertools.combinations(pairs['ticker'], 2))


# TO DO: solve this insanely amount of data problem
# filter more volatile stocks?

# make a dict where keys are the pairs and values are initialized as a dataframe
cointegrated_test = {}
for pair in tqdm(coint_pairs):
    tick1 = data[pair[0]]['Adj Close']
    tick2 = data[pair[1]]['Adj Close']
    info = pd.concat([tick1, tick2], axis=1)
    cointegrated_test[pair] = pd.DataFrame(
        columns=['cointegrated'], index=info.index)
    for i in range(60, len(info)):

        info_60 = info.iloc[:i, :]
        # if statsGen raise an error, skip this iteration
        try:
            result = statsGen(info_60)
        except:
            continue
        if result.lr1[0] > result.cvt[0, 0] and not result:
            cointegrated_test[pair].loc[info.index[i], 'cointegrated'] = True
        else:
            cointegrated_test[pair].loc[info.index[i], 'cointegrated'] = False

with open('cointegrated_test.pkl', 'wb') as f:
    pickle.dump(cointegrated_test, f)
