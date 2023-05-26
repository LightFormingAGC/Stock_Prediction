from concurrent.futures import thread
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import web_scraping as ws
import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


def data_generator(ticker):

    data = pd.DataFrame()
    data['Close'] = wb.DataReader(
        ticker, data_source='yahoo', start='2020-1-1')['Close']
    data['return'] = data['Close'].pct_change()

    # find SMA21
    data['SMA21'] = data['Close'].rolling(window=21).mean()

    data['distance'] = data['Close'] - data['SMA21']
    mean = data['distance'].mean()
    std = data['distance'].std()

    threshold = mean + std * 2

    data['distance'].dropna().plot(figsize=(10, 6), legend=True)
    plt.axhline(threshold, color='red', linestyle='--')
    plt.axhline(-threshold, color='red', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.show()


ticker = input('Enter ticker: ')
data_generator(ticker)
