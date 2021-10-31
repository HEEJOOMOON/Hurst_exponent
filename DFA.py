import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import seaborn as sns


def cumsum(df: pd.Series,
           )-> pd.Series:
    '''

    :param df: (pd.Series) Close price
    :return: (pd.Series) Stochastic cumsum series
    '''

    df = df.pct_change()
    df = df - df.mean()
    df = df.cumsum()
    return df


def detrend(df: pd.Series,
            ) -> pd.Series:
    '''

    :param df: (pd.Series) Stochastic time series cumsum data
    :return: (pd.Series) detrended stock prices
    '''

    X = np.arange(0, len(df))*0.1
    Y = df
    X = sm.add_constant(X)

    results = np.linalg.inv(X.T @ X) @ X.T @ Y

    trend = results[0] + results[1]*X[:, 1]

    return Y - trend


def DFA(data: pd.Series,
        window: list):
    '''

    :param data: (pd.Series) Stochastic time series cumsum data
    :param window: (list) parameter tau
    :return: (np.ndarray) hurst exponent
    '''

    F_2 = []
    size = []
    for w in window:
        for i in range(int(len(data)/w)):
            detrended = detrend(data[i*window:(i+1)*window])
            F_2 += np.sum(np.abs(detrended)**2) / window

        F_2 /= int(len(data)/w)
        hurst = np.log(np.sqrt(F_2)) / np.log(window)

        if hurst > 1 or hurst < 0:
            raise ValueError('Hurst exponent는 0과 1 사이')
        else: return hurst


def time_scale(origin: pd.Series,
               time: list,
               window: list,
               ) -> pd.DataFrame:
    '''

    :param origin: (pd.Series total period's close prices
    :param time: (list)
    :param window: (list)
    :return: (defaultdict) hurst_exponent
    '''

    out = {}
    for t in time:
        for l in range(0, len(origin)-t):
            tmp = cumsum(origin[l:l+t])
            hurst_ = DFA(tmp, window)
            try: out[t]+=[hurst_]
            except KeyError: out[t]=[hurst_]

    return pd.DataFrame.from_dict(out, columns=time)


if __name__=='__main__':
    data = yf.download('KO', '2017-01-01')
    data = data['Close']
    time = [64, 128, 256]
    window = [2, 4, 8, 16]
    results = time_scale(data, time, window)
    sns.heatmap(results)
    plt.show()