# Trend-scanning labeling and Hurst exponent labeling

import numpy as np
import pandas as pd
from Intro import hurst_exponent
from typing import Union


def cal_tval(df: np.array,
             ):
    x = np.ones((df.shape[0], 2))
    x[:, 1] = np.arange(df.shape[0])
    result = np.linalg.lstsq(x, df, rcond=None)
    beta_0, beta_1 = result[0]
    se = np.sqrt(result[1]/(df.shape[0]-2)) / np.sqrt(np.sum((x[:, 1]-np.mean(x[:, 1])**2)))
    tval = beta_1/se

    return tval


def trend_scanning_label(df: Union[pd.Series, np.array],
                         L: list,
                         ):
    '''

    :param df: time sereis data
    :param L: a set of time lag l
    :return: trend-scanning label dataframe
    '''


    return None


def output_label():

    return None

