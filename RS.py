import FinanceDataReader as fdr
import pandas as pd
import numpy as np


def cal_R_S(df):
    z = np.cumsum(df - df.mean())
    R = np.max(z) - np.min(z)
    S = df.std()

    return R/S


def hurst_exponent(df, max_split):
    size = []
    R_S = []

    if len(df) < 30:
        raise ValueError('A time series is too short. Length must be more than 30')

    if max_split >= len(df):
        raise ValueError('max_split must be smaller than length of data')

    for i in range(2, max_split):
        tmp_list = []

        for j in np.array_split(df, i):
            tmp_list.append(cal_R_S(j))
        R_S.append(np.log(np.mean(tmp_list)))
        size.append(np.log(len(j)))

    results = np.polyfit(size, R_S, 1)

    return results[0]


if __name__ == '__main__':
    us500 = fdr.DataReader('US500', '2010-01-01')
    splits = []

