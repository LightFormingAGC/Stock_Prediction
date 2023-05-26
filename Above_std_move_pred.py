import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import math
import yfinance as yf
yf.pdr_override()

def find_volatile_move(ticker, direction, t):

    data = pd.DataFrame()
    data['Close'] = wb.DataReader(
        ticker, data_source='yahoo', start='2013-1-1')['Close']

    if t == 't0':
        data = data[:-1]

    pct_change = data.pct_change()
    log_returns = np.log(1 + pct_change)
    u = log_returns.mean()
    std = log_returns.std()

    # find the z-score of each day log return
    z_score = (log_returns - u) / std
    data['z_score'] = z_score

    direction = direction.lower()
    if direction == 'long':

        # check the average time between log_return above 1.5 std
        std_above_1_5 = []
        c = 0
        for i in range(len(data)):
            if data.z_score[i] < 1.5:
                c += 1
            else:
                std_above_1_5.append(c)
                c = 0
        avg_d = np.mean(std_above_1_5)
        std_d = np.std(std_above_1_5)
        tradable_cond = math.ceil(avg_d + 1.3 * std_d)

        check_period = data[-tradable_cond:]
        if check_period.z_score.max() >= 1.5:
            print(f"Sorry {ticker} is not tradable for long position")
            print(
                f"The log return has been above 1 std for the past {tradable_cond} days")
        else:
            #
            print(f"{ticker} is tradable for long position")
            expected_stop_win = math.e ** (u + 1.5 * std) * data.Close[-1]
            print(
                f"The log return hasn't been above 1.5 std for the past {tradable_cond} days")
            print(
                f"Expected stop win is {round(expected_stop_win[0], 2)}")

    elif direction == 'short':
        # check the average time between log_return below -1.5 std
        std_below_1_5 = []
        c = 0
        for i in range(len(data)):
            if data.z_score[i] > -1.5:
                c += 1
            else:
                std_below_1_5.append(c)
                c = 0
        avg_d = np.mean(std_below_1_5)
        std_d = np.std(std_below_1_5)
        tradable_cond = math.ceil(avg_d + 1.3 * std_d)

        check_period = data[-tradable_cond:]
        if check_period.z_score.min() <= -1.5:
            print(f"Sorry {ticker} is not tradable for short position")
            print(
                f"The log return has been below -1 std for the past {tradable_cond} days")
        else:
            #
            print(f"{ticker} is tradable for short position")
            print(
                f"The log return hasn't been below -1.5 std for the past {tradable_cond} days")
            expected_stop_win = math.e ** (u - 1.5 * std) * data.Close[-1]
            print(
                f"Expected stop win is {round(expected_stop_win[0], 2)}")


ticker, direction, t = input(
    "Enter ticker, direction(long/short) and trading_period(t0/t1): ").split(' ')
find_volatile_move(ticker, direction, t)
