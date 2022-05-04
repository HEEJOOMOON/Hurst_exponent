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

    import sys
    sys.path.append('/home/hjmoon/PycharmProjects/Stochastic_Process/')
    from OU_Process import Ornstein_Uhlenbeck
    from Intro import hurst_exponent
    import matplotlib.pyplot as plt

    simulation = pd.DataFrame()
    simulation['RW'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=1, n=10000)
    simulation['HL1'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=2**(-1/50), n=10000)
    simulation['HL2'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=2**(-1/100), n=10000)
    simulation['HL3'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=2**(-1/200), n=10000)
    L = [30, 100]

    returns = simulation.pct_change()
    t_values = pd.DataFrame()
    t_values['RW'] = trend_scanning_label(returns['RW'], L=L)
    t_values['HL1'] = trend_scanning_label(returns['HL1'], L=L)
    t_values['HL2'] = trend_scanning_label(returns['HL2'], L=L)
    t_values['HL3'] = trend_scanning_label(returns['HL3'], L=L)

