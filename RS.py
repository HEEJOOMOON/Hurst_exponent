import FinanceDataReader as fdr
import numpy as np
from typing import Union
import pandas as pd


def cal_R_S(df: np.array):

    df = np.log(df)
    z = np.cumsum(df - np.mean(df))
    R = np.max(z) - np.min(z)
    S = np.std(df)

    return R/S


def hurst_exponent(df: Union[np.array, pd.Series],
                   max_split: Union[int, list],
                   ):
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
        size.append(np.log(len(j)/2))

    results = np.polyfit(size, R_S, 1)

    if results[0] < 0 or results[0] > 1:
        raise ResultsError

    else: return results[0]

class ResultsError(Exception):
    def __init__(self):
        super().__init__('Hurst exponent must be between 0 and 1.')


if __name__ == '__main__':
    us500 = fdr.DataReader('US500', '2010-01-01', '2012-12-31').Close
    splits = [32]
    for s in splits:
        print('Hurst Exponent with %d splits:' %s, hurst_exponent(us500, max_split=s))