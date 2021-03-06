import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import FinanceDataReader as fdr
import seaborn as sns
from statsmodels.tsa.filters.hp_filter import hpfilter


def cumsum(df: pd.Series,
           )-> pd.Series:
    '''

    :param df: (pd.Series) Close price
    :return: (pd.Series) Stochastic cumsum series
    '''

    df = df.pct_change()
    df = df - df.mean()
    df = df.cumsum()
    return df.dropna()


def detrend(df: pd.Series,
            method: str
            ) -> pd.Series:
    '''

    :param df: (pd.Series) time-serise data
    :param method: 'linear' or 'hp_filter'
    :return: (pd.Series) detrended time series data
    '''

    if method=='linear':
        X = np.arange(0, len(df))*0.1
        Y = df
        X = sm.add_constant(X)

        results = np.linalg.inv(X.T @ X) @ X.T @ Y

        trend = results[0] + results[1]*X[:, 1]

    elif method == 'hp_filter':
        None

    else:
        raise KeyError('The method is linear or hp_filter')

    return Y - trend


def DFA(data: pd.Series,
        window: list):
    '''

    :param data: (pd.Series) Stochastic time series cumsum data
    :param window: (list) parameter tau
    :return: (np.ndarray) hurst exponent
    '''

    F = []
    size = []
    for w in window:
        tmp = 0
        for i in range(int(len(data)/w)):
            detrended = detrend(data[i*w:(i+1)*w])
            tmp += np.sum(np.abs(detrended)**2) / w

        tmp /= int(len(data)/w)
        size.append(np.log(w))
        F.append(np.log(np.sqrt(tmp)))
    hurst = np.polyfit(size, F, 1)[0]

    if hurst > 1 or hurst < 0:
        raise ValueError('Hurst exponent는 0과 1 사이')
    else: return hurst


def time_scale(origin: pd.Series,
               t: int,
               window: list,
               ) -> pd.DataFrame:
    '''

    :param origin: (pd.Series) total period's close prices
    :param time: (int)
    :param window: (list)
    :return: (defaultdict) hurst_exponent
    '''

    out = {}
    for l in range(0, len(origin)-(t+1)):
        tmp = cumsum(origin[l:l+t+1])
        hurst_ = DFA(tmp, window)
        try: out[t]+=[hurst_]
        except KeyError: out[t]=[hurst_]

    return out


if __name__=='__main__':
    data = fdr.DataReader('US500')
    data = data['Close']
    window = [2, 4, 8, 16, 32, 64]
    results = time_scale(data, t=64, window=window)