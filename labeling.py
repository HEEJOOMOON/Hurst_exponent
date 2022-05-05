# Trend-scanning labeling and Hurst exponent labeling

import numpy as np
import pandas as pd
import statsmodels.api as sm1
import warnings
warnings.filterwarnings(action='ignore')

# @ Machine Learning for Asset Managers p.68-69

def cal_tval(close):
    # tValue from a linear trend
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm1.OLS(close, x).fit()
    return ols.tvalues[1]


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
            tmp.loc[df.index[end_t-1]] = cal_tval(df.iloc[idx: end_t])
        dt = tmp.replace([np.inf, -np.inf, np.nan], 0).abs().idxmax()
        out.loc[i, ['t1', 'tVal', 'bin']] = df.index[tmp.index[-1]], tmp[dt], np.sign(tmp[dt])

    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    return out.dropna(subset=['bin'])


if __name__=='__main__':

    import sys
    sys.path.append('/home/hjmoon/PycharmProjects/Stochastic_Process/')
    from OU_Process import Ornstein_Uhlenbeck
    from Intro import hurst_label
    import matplotlib.pyplot as plt
    import seaborn as sns

    simulation = pd.DataFrame()
    simulation['RW'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=1, n=2000)
    simulation['HL1'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=2**(-1/200), n=2000)
    simulation['HL2'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=2**(-1/100), n=2000)
    simulation['HL3'] = Ornstein_Uhlenbeck.simulation(mu=0, sigma=1, theta=2**(-1/50), n=2000)
    L = [30, 100]

    returns = simulation.pct_change()
    t_values = pd.DataFrame()
    t_values['RW'] = trend_scanning_label(simulation['RW'], L=L).tVal
    t_values['HL1'] = trend_scanning_label(simulation['HL1'], L=L).tVal
    t_values['HL2'] = trend_scanning_label(simulation['HL2'], L=L).tVal
    t_values['HL3'] = trend_scanning_label(simulation['HL3'], L=L).tVal

    plt.style.use('seaborn')
    plt.figure(figsize=(16, 4))
    xlim = (-5, 5)
    ax1 = plt.subplot(141)
    ax1.set(xlim=xlim)
    ax1.titl.set_text('Random Walk')
    sns.regplot(y=t_values.RW, x=simulation.RW, scatter_kws={'color': 'blue'}, line_kws={'color':'red'})
    ax2 = plt.subplot(142)
    ax2.set(xlim=xlim)
    ax2.titl.set_text('Half-life: 200')
    sns.regplot(y=t_values.HL1, x=simulation.HL1, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
    ax3 = plt.subplot(143)
    ax3.set(xlim=xlim)
    ax3.titl.set_text('Half-life: 100')
    sns.regplot(y=t_values.HL2, x=simulation.HL2, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
    ax4 = plt.subplot(144)
    ax4.set(xlim=xlim)
    ax4.titl.set_text('Half-life: 50')
    sns.regplot(y=t_values.HL3, x=simulation.HL3, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
    plt.title("Correlation b/w returns and trend-scanning label")
    plt.show()

    # Hurst exponent value labeling

    hurst_values = pd.DataFrame()
    hurst_values['RW'] = hurst_label(simulation['RW'], max_length=L[1])
    hurst_values['HL1'] = hurst_label(simulation['HL1'], max_length=L[1])
    hurst_values['HL2'] = hurst_label(simulation['HL2'], max_length=L[1])
    hurst_values['HL3'] = hurst_label(simulation['HL3'], max_length=L[1])

