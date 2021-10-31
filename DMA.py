import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from DFA import cumsum

def cal_ma(df: pd.Series,
           window: int) -> pd.Series:
    '''
    :param df: (pd.Series) Close prices
    :param window: (int) Rolling window
    :return: (pd.Series) Rolling window close prices
    '''

    return df.rolling(window).mean().dropna()


def DMA(df: pd.Series,
               window: list,
               ) -> np.array:
    '''

    :param df: (pd.Series) Close price
    :param window: (list) Rolling window
    :return: (np.ndarray) hurst exponent
    '''
    F = []
    size = []
    for w in window:
        sigma_MA = (np.sum(df.iloc[w:]-cal_ma(df, w))**2) / w
        F.append(np.log(np.sqrt(sigma_MA)))
        size.append(np.log(w))
    hurst = np.polyfit(size, F, 1)[0]
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

    out = {}
    for t in time:
        for l in range(0, len(origin)-t):
            tmp = cumsum(origin[l:l+t])
            hurst_ = DMA(tmp, window)
            try:
                out[t] += hurst_
            except KeyError:
                out[t] = hurst_
    return pd.DataFrame.from_dict(out, columns=time)


if __name__ == '__main__':
    df = yf.download('KO', '2017-01-01')
    df = df['Close']
    time = [64, 128, 256]
    window = [3, 5, 10, 20, 60, 120]
    out = {}
    hurst_ = time_scale(df, time, window)
    sns.clustermap(hurst_)
    plt.show()