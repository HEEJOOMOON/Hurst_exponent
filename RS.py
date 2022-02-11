import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cal_R_S(df):
    z = np.cumsum(df - df.mean())
    R = np.max(z) - np.min(z)
    S = df.std()

    return R/S


def hurst_exponent(df):
    size = []
    R_S = []

    if len(df) < 30:
        raise ValueError('A time series is too short. Length must be more than 30')

    for i in range(30, len(df/2)):
        tmp_list = []

        for j in np.array_split(df, i):
            tmp_list.append(cal_R_S(j))
        R_S.append(np.log(np.mean(tmp_list)))
        size.append(np.log(len(j)))

    results = np.polyfit(size, R_S, 1)

    return results[0]


def time_series_hurst_expo(n, df, range_):
    '''

    :param n: (int) period of time series
    :param j: (int) time interval
    :param df: (pd.Series) Close price
    :param range_: R/S range
    :return:
    '''
    h_df = pd.DataFrame(columns=['hurst_exp'])
    for i in range(0, len(df)-n):
        h_df.loc[df[i:i+n].index[-1]] = hurst_exponent(range_, df[i:i+n])

    return h_df


if __name__ == '__main__':
    df = fdr.DataReader('US500')
    data = df.Close
    range_ = [2, 4, 8, 16, 32, 64]
    h_df = time_series_hurst_expo(128, data, range_)
