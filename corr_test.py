from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import dcor.independence as dcor


ticker = input("Enter the ticker: ")
data = pd.DataFrame()

# Temperary fix
yf.pdr_override()

data['Close'] = pdr.get_data_yahoo(ticker)['Adj Close']
data['IXIC'] = pdr.get_data_yahoo('QQQ')['Adj Close']
data['SPX'] = pdr.get_data_yahoo('SPY')['Adj Close']
data['DJI'] = pdr.get_data_yahoo('DIA')['Adj Close']

data = data.dropna()

# data['Close'] = wb.DataReader(
#     ticker, data_source='yahoo', start='2017-1-1')['Close']
# data['IXIC'] = wb.DataReader(
#     'QQQ', data_source='yahoo', start='2017-1-1')['Close']
# data['SPX'] = wb.DataReader(
#     'SPY', data_source='yahoo', start='2017-1-1')['Close']
# data['DJI'] = wb.DataReader(
#     'DIA', data_source='yahoo', start='2017-1-1')['Close']
ticker_c = data['Close'].pct_change()
Market_c_IXIC = data['IXIC'].pct_change()
Market_c_SPX = data['SPX'].pct_change()
Market_c_DJI = data['DJI'].pct_change()

ticker_log_return = np.log(1+ticker_c).dropna()
market_log_return_IXIC = np.log(1+Market_c_IXIC).dropna()
market_log_return_SPX = np.log(1+Market_c_SPX).dropna()
market_log_return_DJI = np.log(1+Market_c_DJI).dropna()

corrs_ixic = []
corrs_spx = []
corrs_dji = []

for i in range(10, len(ticker_log_return)):
    corrs_ixic.append(ticker_log_return[:i].corr(market_log_return_IXIC[:i]))
    corrs_spx.append(ticker_log_return[:i].corr(market_log_return_SPX[:i]))
    corrs_dji.append(ticker_log_return[:i].corr(market_log_return_DJI[:i]))


print('')

corr1 = np.mean(corrs_ixic)
corr2 = np.mean(corrs_spx)
corr3 = np.mean(corrs_dji)

# print correlation coefficient and relative performance
print(
    f"P({ticker}, IXIC): {round(corr1, 3)}")

if corr1 < 0.3:
    print("Not following IXIC")
elif corr1 > 0.3 and corr1 < 0.7:
    print("Moderate so trend must be matched")
elif corr1 > 0.7:
    print("Strong so trend and buying cost must match")
plt.hist(corrs_ixic, bins=50)
plt.show()

print(
    f"P({ticker}, SPX): {round(corr2, 3)}")
if corr2 < 0.3:
    print("Not following SPX")
elif corr2 > 0.3 and corr2 < 0.7:
    print("Moderate so trend must be matched")
elif corr2 > 0.7:
    print("Strong so trend and buying cost must match")
plt.hist(corrs_spx, bins=50)
plt.show()

print(
    f"P({ticker}, DJI): {round(corr3, 3)}")

if corr3 < 0.3:
    print("Not following DJI")
elif corr3 > 0.3 and corr3 < 0.7:
    print("Moderate so trend must be matched")
elif corr3 > 0.7:
    print("Strong so trend and buying cost must match")
plt.hist(corrs_dji, bins=50)
plt.show()
