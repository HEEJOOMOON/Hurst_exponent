# Trend-scanning labeling and Hurst exponent labeling

import numpy as np
import pandas as pd


def cal_tval(df: np.array,
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
            end_t = idx+l-1
            tmp.cal_tval(df.iloc[idx:end_t])
            tmp.loc[end_t] = cal_tval(df.iloc[idx: end_t])
        dt = tmp.replace([np.inf, -np.inf, np.nan], 0).abs().idxmax()
        out.loc[i, ['t1', 'tVal', 'bin']] = tmp.index[-1], tmp[dt], np.sign(tmp[dt])

    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'])
    return out.dropna(subset='bin')


if __name__=='__main__':
    from Intro import hurst_exponent
