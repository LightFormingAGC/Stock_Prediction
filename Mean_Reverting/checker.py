
import warnings
import os
from tqdm.auto import tqdm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import math
from pandas_datareader import data as wb
import numpy as np
import pandas as pd
from inspect import trace
import yfinance as yf
import pickle
import itertools
# pandas_market_calendars
from pandas_market_calendars.exchange_calendar_nyse import NYSEExchangeCalendar

import warnings
warnings.filterwarnings('ignore')


# look ahead bias, fix it


yf.pdr_override()
warnings.filterwarnings('ignore')


data = pd.read_csv('cointegrated_pairs.csv')


def statsGen(data):

    adjClose = data.iloc[:, :2]
    result = coint_johansen(adjClose, det_order=0, k_ar_diff=1)

    return result
    return result.lr1, result.cvt, result.lr2, result.cvm, result.evec


# lim t choose: 5 years to 15 years
# threshold: 80 pct to 99 pct, increment: 1 pct
log_rts = {}
for i in np.arange(0.005, 0.1, 0.005):
    for waiting in np.arange(260 * 5, 260 * 15, 260):
        log_rts[(i, waiting)] = []


def coint_gen(dates, wait):

    # checking conditions
    pairs = pd.read_csv('cointegrated_pairs.csv', index_col=False)
    # long run expectation, wait for four trading years
    # open pkl

    with open('datas.pkl', 'rb') as f:
        data = pickle.load(f)

    output = []
    for pair in pairs['ticker']:

        info = data[pair].loc[:dates]

        if len(info) < wait:
            continue

        if info.index[0] > pd.to_datetime(dates):
            continue

        try:
            todays_data = info.loc[dates]
        except:
            continue
        avg_volume = info['Volume'].mean()
        avg_transaction = info['transaction amount'].mean()

        if todays_data['Volume'] < 4000000 or \
                todays_data['Adj Close'] < 5 or \
                todays_data['transaction amount'] < 40000000 or \
                avg_volume < 4000000 or \
                avg_transaction < 40000000:
            continue
        else:
            output.append(pair)

    # get passed pairs
    coint_pairs = list(itertools.combinations(output, 2))
    return coint_pairs


def PnL(coint_pairs, dates, log_rts, waiting):

    with open('datas.pkl', 'rb') as f:
        data = pickle.load(f)

    # check for cointegration
    for cp in coint_pairs:

        tick1 = data[cp[0]].loc[:dates]
        tick2 = data[cp[1]].loc[:dates]
        tick1prices = tick1['Adj Close']
        tick2prices = tick2['Adj Close']

        try:
            stats = statsGen(
                pd.concat([tick1prices, tick2prices], axis=1))
        except:
            continue

        # 95 pct threshold for lr and mle
        if stats.lr1[0] > stats.cvt[0][1] or stats.lr2[0] > stats.cvm[0][1]:
            evec = stats.evec
            coef1 = evec[:, 0][0]
            coef2 = evec[:, 0][1]
        else:
            continue

        portfolio = np.array(coef1 * tick1prices + coef2 * tick2prices)

        # 负数问题

        today_value = portfolio[-1]

        for param in log_rts.keys():

            thresh = param[0]

            if (portfolio > today_value).sum() / len(portfolio) < thresh:
                signal = -1
            elif (portfolio < today_value).sum() / len(portfolio) < thresh:
                signal = 1
            else:
                signal = 0

            dates_idx_1 = tick1.index.get_loc(dates)
            dates_idx_2 = tick2.index.get_loc(dates)

            tick1_tmr = data[cp[0]].iloc[dates_idx_1 + 1, :]
            tick2_tmr = data[cp[1]].iloc[dates_idx_2 + 1, :]

            # 0.95 for transaction cost, spread, failed order etc
            if signal == 1:
                cost = 0
                gains = 0
                if coef1 < 0:
                    cost += tick1_tmr['Adj Close'] * (-coef1)
                    gains += tick1prices[-1] * (-coef1)
                else:
                    cost += tick1prices[-1] * coef1
                    gains += tick1_tmr['Adj Close'] * coef1
                if coef2 < 0:
                    cost += tick2_tmr['Adj Close'] * (-coef2)
                    gains += tick2prices[-1] * (-coef2)
                else:
                    cost += tick2prices[-1] * coef2
                    gains += tick2_tmr['Adj Close'] * coef2

                log_rts[(thresh, waiting)].append(
                    math.log(gains * 0.99 / cost))

            elif signal == -1:
                cost = 0
                gains = 0
                if coef1 < 0:
                    cost += tick1prices[-1] * (-coef1)
                    gains += tick1_tmr['Adj Close'] * (-coef1)
                else:
                    cost += tick1_tmr['Adj Close'] * coef1
                    gains += tick1prices[-1] * coef1
                if coef2 < 0:
                    cost += tick2prices[-1] * (-coef2)
                    gains += tick2_tmr['Adj Close'] * (-coef2)
                else:
                    cost += tick2_tmr['Adj Close'] * coef2
                    gains += tick2prices[-1] * coef2

                log_rts[(thresh, waiting)].append(
                    math.log(gains * 0.99 / cost))

        # save log_rt
    with open('log_rt_backtest.pkl', 'wb') as f:
        pickle.dump(log_rts, f)

    return log_rts


# backtest end date must be one day before today
end = '2023-05-24'
# start is two yerars before today
out_sample_start = pd.to_datetime(end) - pd.DateOffset(years=4)
out_sample_start = out_sample_start.strftime('%Y-%m-%d')
in_sample_start = pd.to_datetime(end) - pd.DateOffset(years=20)
in_sample_start = in_sample_start.strftime('%Y-%m-%d')


# get dates when the market is open
nyse = NYSEExchangeCalendar()
trading_days = nyse.valid_days(
    start_date=in_sample_start, end_date=out_sample_start)
trading_days = pd.to_datetime(trading_days).strftime('%Y-%m-%d')

for day in tqdm(trading_days):
    for param in log_rts.keys():
        thresh = param[0]
        waiting = param[1]
        log_rts = PnL(coint_gen(day, waiting), day, log_rts, waiting)


# print(simulating_trading('2023-05-19'))


# def portfolio_builder(pair):

#     data = wb.DataReader(pair, data_source='yahoo', interval='1d')[
#         ['Adj Close']]

#     data = data.dropna()

#     data[f'coef_{pair[0]}'] = None
#     data[f'coef_{pair[1]}'] = None
#     data['portfolio'] = None

#     for i in tqdm(range(50, len(data))):

#         evec = statsGen(data.iloc[:i+1, :])[4][0]
#         coef1 = evec[0]
#         coef2 = evec[1]
#         data.iloc[i, -3] = coef1
#         data.iloc[i, -2] = coef2

#         if i != len(data)-1:
#             data.iloc[i+1, -1] = coef1 * \
#                 data.iloc[i, 0] + coef2 * data.iloc[i, 1]

#     adjdata = data.dropna()
#     adjdata['rolling_mean'] = adjdata['portfolio'].expanding().mean()

#     check = buyorsell(adjdata)

#     if check == 1:
#         print(f'Portfolio Coefficents for {pair} are {coef1} and {coef2}')
#         print('You should long the portfolio')
#     elif check == -1:
#         print(f'Portfolio Coefficents for {pair} are {coef1} and {coef2}')
#         print('You should short the portfolio')
#     else:
#         pass


# for i in tqdm(range(len(data))):
#     pair = tuple(data.iloc[i, :2])
#     backTest(pair)

# print("Negative Coefficents mean the portfolio is shorting the stock")
# print("Positive Coefficents mean the portfolio is long the stock")