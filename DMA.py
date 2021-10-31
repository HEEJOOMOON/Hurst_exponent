import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from collections import defaultdict

def cal_ma(df: pd.Series,
           window: int) -> pd.Series:
    '''
    :param df: (pd.Series) Close prices
    :param window: (int) Rolling window
    :return: (pd.Series) Rolling window close prices
    '''

    return df.rolling(window).mean().dropna()


def hurst_expo(df: pd.Series,
               window: int,
               ) -> np.array:
    '''

    :param df: (pd.Series) Close price
    :param window: (int) Rolling window
    :return: (np.ndarray) hurst exponent
    '''

    sigma_MA = (1/(len(df)-window))*(np.sum(df.iloc[window:]-cal_ma(df, window))**2)
    hurst = np.log(np.sqrt(sigma_MA)) / np.log(window)
    return hurst


def time_scale(origin: pd.Series,
               time: list,
               window: list,
               ) -> pd.DataFrame:
    '''

    :param origin: (pd.Series) total period's close prices
    :param time: (list)
    :param window: (list)
    :return: (pd.DataFrame) hurst_exponent (index: time, column: scale)
    '''

    out = defaultdict(dict)
    for t in time:
        for l in range(0, len(origin)-t):
            tmp = origin[l:l+t]
            for w in window:
                hurst_ = hurst_expo(tmp, w)
                try:
                    out[t] += hurst_
                except KeyError:
                    out[t] = hurst_
    return out


if __name__ == '__main__':
    df = yf.download('KO', '2000-01-01')
    df = df['Close']
    time = [64, 128, 256]
    window = [20, 60, 120]
    out = {}
    hurst_ = time_scale(df, time, window)
    sns.clustermap(hurst_)
    plt.show()