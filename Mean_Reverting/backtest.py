
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

import warnings
warnings.filterwarnings('ignore')


# cointegration 本质: y-ax = I(0)
# we need to find coefficients that are constant over time to ensure revert back to the mean when mispriced
# most cointegration pass because coefficients changes, not because they are stationary
# for two prices to be cointegraed, when one is mispriced, it has to revert back becaus etherwise mean changes. (under same coefficient)


yf.pdr_override()


data = pd.read_csv('cointegrated_pairs.csv')


def statsGen(data):

    adjClose = data.iloc[:, :2]
    result = coint_johansen(adjClose, det_order=0, k_ar_diff=1)

    return result
    return result.lr1, result.cvt, result.lr2, result.cvm, result.evec


# ignore asymptotic behavior, just use 60 month waiting time for stationary check
# threshold: 90 pct to 99 pct, increment: 1 pct
log_rts = {}
for i in np.arange(0.005, 0.05, 0.005):
    log_rts[i] = []


def coint_gen(dates):

    # checking conditions
    pairs = pd.read_csv('cointegrated_pairs.csv', index_col=False)
    # long run expectation, wait for four trading years
    # open pkl

    with open('datas.pkl', 'rb') as f:
        data = pickle.load(f)

    output = []
    for pair in pairs['ticker']:

        info = data[pair].loc[:dates]

        if len(info) < 60:
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


def pass_test(results):
    if results.lr1[0] > results.cvt[0, 0]:
        return True
    return False


def PnL(coint_pairs, dates, log_rts):

    with open('datas.pkl', 'rb') as f:
        data = pickle.load(f)

    # check for cointegration
    for cp in coint_pairs:

        tick1 = data[cp[0]].loc[:dates]
        tick2 = data[cp[1]].loc[:dates]
        tick1prices = tick1['Adj Close']
        tick2prices = tick2['Adj Close']

        df = pd.concat([tick1prices, tick2prices], axis=1)
        df.columns = [cp[0], cp[1]]
        df = df.dropna()

        # check passed pct
        passed = []
        mos_recent_evec = [0, 0]
        for i in range(60, len(df)):
            history = df.iloc[:i, :]
            results = coint_johansen(history, det_order=0, k_ar_diff=1)
            if pass_test(results):
                # update most recent evec
                mos_recent_evec = results.evec[:, 0]
                passed.append(True)
            else:
                passed.append(False)

        # check if pct pass
        pct = passed.mean()
        if not pct > 0.8:
            continue

        coef1 = mos_recent_evec[0]
        coef2 = mos_recent_evec[1]

        # build the portfolio
        portfolio = df[cp[0]] * coef1 + df[cp[1]] * coef2

        # 负数问题

        today_value = portfolio[-1]

        for param in log_rts.keys():

            thresh = param

            if (portfolio > today_value).sum() / len(portfolio) < thresh:
                signal = -1
            elif (portfolio < today_value).sum() / len(portfolio) < thresh:
                signal = 1
            else:
                signal = 0

            dates_idx1 = data[cp[0]].index.get_loc(dates)
            dates_idx2 = data[cp[1]].index.get_loc(dates)

            tick1_tmr = data[cp[0]].iloc[dates_idx1 + 1]
            tick2_tmr = data[cp[1]].iloc[dates_idx2 + 1]

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

                log_rts[thresh].append(math.log(gains * 0.99 / cost))

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

                log_rts[thresh].append(math.log(gains * 0.99 / cost))

    return log_rts


# backtest end date must be one day before today
end = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
# start is two yerars before today
out_sample_start = pd.to_datetime(end) - pd.DateOffset(years=5)
out_sample_start = out_sample_start.strftime('%Y-%m-%d')
in_sample_start = pd.to_datetime(end) - pd.DateOffset(years=30)
in_sample_start = in_sample_start.strftime('%Y-%m-%d')


# get dates when the market is open
nyse = NYSEExchangeCalendar()
trading_days = nyse.valid_days(
    start_date=in_sample_start, end_date=out_sample_start)
trading_days = pd.to_datetime(trading_days).strftime('%Y-%m-%d')

for day in tqdm(trading_days):
    log_rts = PnL(coint_gen(day), day, log_rts)

# save log_rt
with open('log_rt_backtest_99clcoint.pkl', 'wb') as f:
    pickle.dump(log_rts, f)
