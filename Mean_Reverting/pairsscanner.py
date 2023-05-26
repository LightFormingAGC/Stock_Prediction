# 95 pct dates are stationary
# 95 pct threshold
# Ergodic Theorem
import time
import warnings
from tqdm.auto import tqdm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import math
from pandas_datareader import data as wb
import numpy as np
import pandas as pd
from inspect import trace
import multiprocessing as mp
import yfinance as yf
yf.pdr_override()
warnings.filterwarnings('ignore')

pairs_info = pd.DataFrame(
    columns=['symb1', 'symb2', 'coef1', 'coef2', 'direction'])


def statsGen(data):

    adjClose = data.iloc[:, :2]
    result = coint_johansen(adjClose, det_order=0, k_ar_diff=1)

    return result.lr1, result.cvt, result.lr2, result.cvm, result.evec


def signalgenerator(row):

    symb1 = row['symb1']
    symb2 = row['symb2']
    pair = (symb1, symb2)

    data = wb.get_data_yahoo(pair)[['Adj Close', 'Volume']]

    data = data.dropna()

    data[f'coef_{pair[0]}'] = None
    data[f'coef_{pair[1]}'] = None
    data['portfolio'] = None

    for i in tqdm(range(50, len(data))):
        try:
            evec = statsGen(data.iloc[:i+1, :])[4][0]
            coef1 = evec[0]
            coef2 = evec[1]
            data.iloc[i, -3] = coef1
            data.iloc[i, -2] = coef2

            if i != len(data)-1:
                data.iloc[i+1, -1] = coef1 * \
                    data.iloc[i, 0] + coef2 * data.iloc[i, 1]
        except:
            pass

    adjdata = data.dropna()

    historical = adjdata.iloc[:-1, :]
    today_value = adjdata.iloc[-1, -1]

    above = historical[historical['portfolio'] >=
                       today_value].shape[0] / historical.shape[0]
    below = historical[historical['portfolio'] <=
                       today_value].shape[0] / historical.shape[0]

    historica_volume = historical['Volume'].mean()

    if above < 0.025 and historica_volume[0] > 3500000 and historica_volume[1] > 3500000:
        return {'symb1': symb1, 'symb2': symb2,
                'coef1': adjdata.iloc[-1, -3], 'coef2': adjdata.iloc[-1, -2], 'direction': 'Short'}
    elif below < 0.025 and historica_volume[0] > 3500000 and historica_volume[1] > 3500000:
        return {'symb1': symb1, 'symb2': symb2,
                'coef1': adjdata.iloc[-1, -3], 'coef2': adjdata.iloc[-1, -2], 'direction': 'Long'}
    else:
        return None


combined = pd.DataFrame()
for j in range(0, 1):
    data = pd.read_csv('cointegrated_pairs_{}.csv'.format(j))
    combined = combined.append(data, ignore_index=True)


if __name__ == '__main__':
    
    start_time = time.time()

    results = []
    with mp.Pool(mp.cpu_count()) as pool:

        for i in tqdm(range(combined.shape[0])):
            results.append(pool.apply_async(
                signalgenerator, args=(combined.iloc[i, :],)))
        for result in results:
            if result.get() != None:
                pairs_info = pairs_info.append(result.get(), ignore_index=True)
    pool.close()
    pool.join()

    print("--- %s seconds ---" % (time.time() - start_time))
