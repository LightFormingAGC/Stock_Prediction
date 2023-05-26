from multiprocessing.resource_sharer import stop
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
from datetime import date


def monte_carlo(ticker, Open_Close, num_simulations=1000):

    # find parameters

    # number of days from start_day to today
    num_days = 1
    data = pd.DataFrame()
    Open_Close = Open_Close.capitalize()

    ###
    yf.pdr_override()
    data[Open_Close] = wb.get_data_yahoo(ticker, start='2013-1-1')[Open_Close]
    # data[Open_Close] = wb.DataReader(
    #     ticker, data_source='yahoo', start='2013-1-1')[Open_Close]
    if Open_Close == 'Open':
        real_time = float(input("What is the today's open price? "))
        data.iloc[-1, 0] = real_time

    pct_change = data.pct_change()
    log_returns = np.log(1 + pct_change)
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    std = log_returns.std()

    # convert start_day to datetime64
    today = date.today().strftime("%Y-%m-%d")
    start_price = data.loc[today]
    wiener = np.random.standard_normal((num_simulations, 1))
    arr = np.zeros_like(num_simulations*(num_days+1),
                        shape=(num_simulations, num_days+1))
    arr[:, 0] = start_price
    for si in range(num_simulations):
        arr[si, 1] = arr[si, 0] * \
            np.exp(drift * num_days + std * wiener[si, 0])

    # reverse the shape
    arr = arr.T

    last_days = arr[-1]
    stop_win = last_days.mean()
    price_now = round(data.iloc[-1].values[0], 2)

    long_short = input("Long or Short: ").capitalize()
    if long_short == "Long":

        print('\n')
        prob = round(1 - (last_days < price_now).mean(), 2)
        print("Probability of making money:", prob)

        print('Entry Price: ', price_now)
        print("Stop win price:", stop_win)
        stop_loss = round(price_now - (stop_win - price_now) * 0.2, 2)
        print("Stop loss price:", stop_loss)
        print('\n')

    elif long_short == "Short":

        print('\n')
        prob = round(1 - (last_days > price_now).mean(), 2)
        print("Probability of making money:", prob)

        print('Entry price:', price_now)
        print("Stop win price:", stop_win)
        stop_loss = round(price_now + (price_now - stop_win) * 0.2, 2)
        print("Stop loss price:", stop_loss)
        print('\n')

    else:
        print("Wrong input")


ticker = input("Enter ticker: ").upper()
Open_Close = input("Open or Close price simulation: ").capitalize()
monte_carlo(ticker, Open_Close)
