
from tqdm.auto import tqdm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import math
from pandas_datareader import data as wb
import numpy as np
import pandas as pd
from inspect import trace
import yfinance as yf
yf.pdr_override()


def trend(data):

    data.loc[:, 'Bollinger'] = data['Adj Close'].rolling(window=20).mean()
    data.loc[:, 'Bollinger Upper'] = data['Bollinger'] + \
        2 * data['Adj Close'].rolling(window=20).std()
    data.loc[:, 'Bollinger Lower'] = data['Bollinger'] - \
        2 * data['Adj Close'].rolling(window=20).std()

    data = data.dropna()

    # for each row, find its all historical log returns and calculate the mean
    data['globalLogReturn'] = data.apply(lambda x: np.log(
        data['Adj Close'].loc[:x.name].pct_change() + 1).mean(), axis=1)

    def short_term_peaks_lows(df):
      # Add a column to the dataframe to store the result
        df['short_term_peak_low'] = ''

        # Iterate over the rows of the dataframe
        for i, row in df.iterrows():
            # Get the date of the current row
            date = row.name
            # Get the opening price of the current row
            opening_price = row['Adj Open']
            closing_price = row['Adj Close']

            potential_peak = max(opening_price, closing_price)
            potential_low = min(opening_price, closing_price)

            # Get the prices for the next 10 days
            next_10_days = df[df.index > date][:2]
            last_10_days = df[df.index < date][-2:]

            # Determine if the current date is a short term peak or low
            is_peak = True
            is_low = True
            for _, next_day in next_10_days.iterrows():
                if next_day['Adj Close'] > potential_peak or next_day['Adj Open'] > potential_peak:
                    is_peak = False
                if next_day['Adj Close'] < potential_low or next_day['Adj Open'] < potential_low:
                    is_low = False

            for _, last_day in last_10_days.iterrows():
                if last_day['Adj Close'] > potential_peak or last_day['Adj Open'] > potential_peak:
                    is_peak = False
                if last_day['Adj Close'] < potential_low or last_day['Adj Open'] < potential_low:
                    is_low = False

                # Set the value in the 'short_term_peak_low' column
            if is_peak:
                df.at[date, 'short_term_peak_low'] = 'peak'
            elif is_low:
                df.at[date, 'short_term_peak_low'] = 'low'
            else:
                df.at[date, 'short_term_peak_low'] = 'neither'

        return df

    data = short_term_peaks_lows(data)

    upinterval = []
    tuple = []

    for i in range(len(data)):
        if data.iloc[i]['short_term_peak_low'] == 'low' and len(tuple) == 0:
            tuple.append(data.index[i])
        elif data.iloc[i]['short_term_peak_low'] == 'low' and len(tuple) == 1:
            tuple[0] = data.index[i]
        elif data.iloc[i]['short_term_peak_low'] == 'peak' and len(tuple) == 1:
            tuple.append(data.index[i])
            upinterval.append(tuple)
            tuple = []
        else:
            continue

    upintervalsReturn = []
    for interval in upinterval:
        upintervalsReturn.append(
            np.log(data['Adj Close'].loc[interval[0]:interval[1]].pct_change() + 1).mean()
        )

    upintervalsReturnAvg = np.mean(upintervalsReturn)
    upintervalsReturnStd = np.std(upintervalsReturn)

    def find_interval(entry, upinterval):
        for interval in reversed(upinterval):
            if entry >= interval[0]:
                if entry < interval[1]:
                    return [interval[0], entry]
                else:
                    return interval
        return None

    for i, row in data.iterrows():

        interval = find_interval(i, upinterval)

        if not interval:
            continue
        if i == interval[0]:
            timedelta = pd.Timedelta(weeks=1)
            tempData = data.loc[interval[0] - timedelta: interval[0], :]
        else:
            tempData = data.loc[interval[0] - timedelta:i, :]

        avgReturn = np.log(tempData['Adj Close'].pct_change() + 1).mean()
        intervalGrowth = np.log(
            data['Adj Close'].loc[interval[0]:interval[1]].pct_change() + 1).mean()

        cond1Up = avgReturn > data['globalLogReturn'][i]
        cond2Up = intervalGrowth < upintervalsReturnAvg + 1 * upintervalsReturnStd
        cond3Up = (min(data['Adj Close'][i], data['Adj Open'][i]) >= data['Bollinger'][i] * 0.98
                   and max(data['Adj Close'][i], data['Adj Open'][i]) >= data['Bollinger'][i])
        cond4Up = (data['Adj Low'][i]/data['Bollinger'][i] - 1 >
                   (data['Bollinger Lower'][i]/data['Bollinger'][i] - 1)*0.5)

        if (cond1Up and cond2Up and cond3Up and cond4Up):
            data.at[i, 'trend'] = 1

    return data
