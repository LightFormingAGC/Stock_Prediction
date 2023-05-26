from numpy.random import randn
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()

# Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/

# temp sol
ticker = input("Enter the ticker: ")
data = pd.DataFrame()
data['Close'] = wb.get_data_yahoo(ticker, start='2017-1-1')['Close']


def hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts

    Parameters
    ----------
    ts : `numpy.array`
        Time series upon which the Hurst Exponent will be calculated

    Returns
    -------
    'float'
        The Hurst Exponent from the poly fit output
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # plot out the different lags
    plt.plot(log(lags), log(tau), 'o')
    
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


test_statistic = hurst(data.Close.values)


print("Hurst(%s):  %s" % (ticker, test_statistic))

if test_statistic < 0.485:
    print("The time series is significant mean reverting")
elif test_statistic > 0.515:
    print("The time series is trending")
else:
    print("The time series significantly followes a Geometric Brownian Motion")
