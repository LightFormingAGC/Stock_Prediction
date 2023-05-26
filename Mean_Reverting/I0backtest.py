# 95 pct dates are stationary
# 95 pct threshold
# Ergodic Theorem
import warnings
from tqdm.auto import tqdm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import math
from pandas_datareader import data as wb
import numpy as np
import pandas as pd
from inspect import trace
import yfinance as yf
import Cointegrated_Paris_Generator as cpg

yf.pdr_override()
warnings.filterwarnings('ignore')

All_returns = []


def statsGen(data):

    adjClose = data.iloc[:, :2]
    result = coint_johansen(adjClose, det_order=0, k_ar_diff=1)

    return result.lr1, result.cvt, result.lr2, result.cvm, result.evec


def buy_sell(data):

    # long run expectation, wait for four trading years

    historical = data.iloc[:-1, :]
    today_value = data.iloc[-1, -3]

    above = historical[historical['portfolio'] >=
                       today_value].shape[0] / historical.shape[0]
    below = historical[historical['portfolio'] <=
                       today_value].shape[0] / historical.shape[0]

    if above < 0.025:
        return -1
    elif below < 0.025:
        return 1
    else:
        return 0

    return data


def backTest(pair):

    data = wb.get_data_yahoo(pair)[['Adj Close', 'Volume']]

    data = data.dropna()

    data[f'coef_{pair[0]}'] = None
    data[f'coef_{pair[1]}'] = None
    data['portfolio'] = None

    for i in tqdm(range(50, len(data))):
        try:
            evec = statsGen(data.iloc[:i+1, :])[4][0]
            coef1 = evec[0]
            coef2 = evec[1]
            data.iloc[i, -3] = coef1
            data.iloc[i, -2] = coef2

            if i != len(data)-1:
                data.iloc[i+1, -1] = coef1 * \
                    data.iloc[i, 0] + coef2 * data.iloc[i, 1]
        except:
            pass

    adjdata = data.dropna()
    adjdata['rolling_mean'] = adjdata['portfolio'].expanding().mean()

    buy_sell = buy_sell(adjdata)

    # temp = adjdata.copy()

    # shift_term = abs(min(temp['portfolio']) - 20)
    # temp['portfolio'] += shift_term
    # temp['rolling_mean'] += shift_term

    # available_capital = 100000
    # num_shares = 0
    # floating_portfolios_value = []

    # for i in range(len(temp)):

    #     # exiting the next day
    #     if num_shares > 0:
    #         available_capital += temp.iloc[i, -3] * num_shares * 0.9
    #     elif num_shares < 0:
    #         available_capital += temp.iloc[i, -3] * num_shares * 1.1

    #     # opening position
    #     # longing
    #     if temp.iloc[i, -1] == 1:

    #         if num_shares == 0:

    #             num_shares = available_capital // temp.iloc[i, -3]
    #             available_capital -= num_shares * temp.iloc[i, -3]

    #     # shorting
    #     elif temp.iloc[i, -1] == -1:

    #         if num_shares == 0:

    #             num_shares = - available_capital // temp.iloc[i, -3]
    #             available_capital -= num_shares * temp.iloc[i, -3]

    #     floating_portfolios_value.append(
    #         available_capital + num_shares * temp.iloc[i, -3])

    # port_return = (floating_portfolios_value[-1] - 100000) / 100000
    # return port_return


sector_ret = {}


for j in range(8, 9):
    All_returns = []
    data = pd.read_csv('cointegrated_pairs_{}.csv'.format(j))
    for i in tqdm(range(len(data))):
        pair = (data['symb1'][i], data['symb2'][i])
        All_returns.append(backTest(pair))

    sector_ret[j] = (np.mean(All_returns), len(
        [i for i in All_returns if i < 0]) / len(All_returns))

# save the result
with open('sector_ret_9.txt', 'w') as f:
    for key, value in sector_ret.items():
        f.write('%s:%s' % (key, value))
        f.write('\n')
