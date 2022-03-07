# Trend-scanning labeling and Hurst exponent labeling

import numpy as np
import pandas as pd
from Intro import hurst_exponent
from typing import Union


def cal_tval(df: np.array,
             ):
    x = np.ones((df.shape[0], 2))
    x[:, 1] = np.arange(df.shape[0])
    result = np.linalg.lstsq()
    beta = result[0]
    se = result[1]
    tval = beta/se

    return tval


def trend_scanning_label(df: Union[pd.Series, np.array],
                         L: list,
                         ):


    return None


def output_label():

    return None

