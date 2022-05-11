import numpy as np
import pandas as pd
import FinanceDataReader as fdr
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
        if idx+max(lags) >= df.shape[0]: continue

        for l in lags:
            end_t = idx+l
            tmp.loc[df.index[end_t]] = cal_tval(df.iloc[idx: end_t])
        dt = tmp.replace([np.inf, -np.inf, np.nan], 0).abs().idxmax()
        out.loc[i, ['t1', 'tVal', 'bin']] = tmp.index[-1], tmp[dt], np.sign(tmp[dt])

    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    return out.dropna(subset=['bin'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    L = [20, 90]
    
    se = fdr.DataReader('005930', '2000-01-01', '2021-12-31').Close
    ts = trend_scanning_label(se, L=L)
    returns = se.pct_change(L[1])
    tmp = pd.DataFrame([ts.tVal.values, ts.bin.values, returns[88:]])
    df_se = tmp.T
    df_se.index = ts.t1
    df_se.columns = ['tVal', 'bin', 'returns']
    
    df_se.bin.loc[df_se.bin == -1.0] = 0

    hynix = fdr.DataReader('000660', '2000-01-01', '2021-12-31').Close
    ts = trend_scanning_label(hynix, L=L)
    returns = hynix.pct_change(L[1])
    tmp = pd.DataFrame([ts.tVal.values, ts.bin.values, returns[88:]])
    df_hy = tmp.T
    df_hy.index = ts.t1
    df_hy.columns = ['tVal', 'bin', 'returns']
    df_hy.bin.loc[df_hy.bin == -1.0] = 0

    naver = fdr.DataReader('035420', '2000-01-01', '2021-12-31').Close
    ts = trend_scanning_label(naver, L=L)
    returns = naver.pct_change(L[1])
    tmp = pd.DataFrame([ts.tVal.values, ts.bin.values, returns[88:]])
    df_na = tmp.T
    df_na.index = ts.t1
    df_na.columns = ['tVal', 'bin', 'returns']
    df_na.bin.loc[df_na.bin == -1.0] = 0

    kakao = fdr.DataReader('035720', '2000-01-01', '2021-12-31').Close
    ts = trend_scanning_label(kakao, L=L)
    returns = kakao.pct_change(L[1])
    tmp = pd.DataFrame([ts.tVal.values, ts.bin.values, returns[88:]])
    df_ka = tmp.T
    df_ka.index = ts.t1
    df_ka.columns = ['tVal', 'bin', 'returns']
    df_ka.bin.loc[df_ka.bin == -1.0] = 0
