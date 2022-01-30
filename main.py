import time
import datetime
import pandas as pd


# Pulling the past 2 years data of the ticker using daily intervals

def pull_data(ticker):
    today = datetime.date.today().timetuple()
    start_time = (datetime.date.today() - datetime.timedelta(days=2000)).timetuple()
    today_s = int(time.mktime(datetime.datetime(today[0], today[1], today[2], 23, 59).timetuple()))
    start_time_s = int(time.mktime(datetime.datetime(start_time[0], start_time[1], start_time[2], 23, 59).timetuple()))

    scrap_site = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}' \
                 f'?period1={start_time_s}&period2={today_s}' \
                 f'&interval=1wk&events=history&includeAdjustedClose=true'
    data = pd.read_csv(scrap_site)
    return data
