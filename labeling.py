# Trend-scanning labeling and Hurst exponent labeling

import numpy as np
import pandas as pd


def cal_tval(df: pd.Series,
             ):
    x = np.ones((df.shape[0], 2))
    x[:, 1] = np.arange(df.shape[0])
    result = np.linalg.lstsq(x, df, rcond=None)
    beta_0, beta_1 = result[0]
    se = np.sqrt(result[1]/(df.shape[0]-2)) / np.sqrt(np.sum((x[:, 1]-np.mean(x[:, 1])**2)))
    tVal = beta_1/se

    return tVal


def trend_scanning_label(df: pd.Series,
                         L: list,
                         ):
    '''

    :param df: time sereis data
    :param L: a set of time lag l (start, end, step)
    :return: trend-scanning label dataframe
    '''

    out = pd.DataFrame(index=df.index, columns=['t1', 'tVal', 'bin'])
    lags = range(*L)
    for i in df.index:
        tmp = pd.Series()
        idx = df.index.get_loc(i)
        if idx+max(lags) > df.shape[0]: continue

        for l in lags:
            end_t = idx+l
            tmp.cal_tval(df.iloc[idx:end_t])
            tmp.iloc[end_t-1] = cal_tval(df.iloc[idx: end_t])
        dt = tmp.replace([np.inf, -np.inf, np.nan], 0).abs().idxmax()
        out.loc[i, ['t1', 'tVal', 'bin']] = df.index[tmp.index[-1]], tmp[dt], np.sign(tmp[dt])

    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    return out.dropna(subset='bin')


if __name__=='__main__':
    from Intro import hurst_exponent
    import matplotlib.pyplot as plt
    import FinanceDataReader as fdr

    df = fdr.DataReader('us500', '2017-01-01').Close
    ts = trend_scanning_label(df, [5, 30])
