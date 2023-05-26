from bs4 import BeautifulSoup
import requests
import datetime
import time
import numpy as np


def scrap_data(ticker):
    today = datetime.date.today().timetuple()
    start_date = (datetime.date.today() - datetime.timedelta(days=730)).timetuple()

    today_s = int(time.mktime(datetime.datetime(today[0], today[1], today[2], 23, 59).timetuple()))
    start_time_s = int(time.mktime(datetime.datetime(start_date[0], start_date[1], start_date[2], 23, 59).timetuple()))
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36'
    }

    params = (
        ('period1', str(start_time_s)),
        ('period2', str(today_s)),
        ('interval', '1d'),
        ('filter', 'history'),
        ('frequency', '1d'),
        ('includeAdjustedClose', 'true'),
    )

    response = requests.get(f'https://finance.yahoo.com/quote/{ticker.upper()}/history', headers=headers, params=params).text
    soup = BeautifulSoup(response, 'html.parser')
    contents = soup.find_all('tr', class_='BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)')

    closing_price = []
    for content in contents:
        price = content.find_all('td', class_='Py(10px) Pstart(10px)')
        try:
            closing_price.append(price[3].text)
        except IndexError:
            pass
    return closing_price
