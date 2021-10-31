import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cal_return(df):
    return df.pct_change().dropna()


def cal_R_S(df):
    z = np.cumsum(df - df.mean())
    R = np.max(z) - np.min(z)
    S = df.std()

    return R/S


def hurst_exponent(range_, df):
    size = []
    R_S = []

    for i in range_:
        tmp_list = []

        for j in np.array_split(df, i):
            tmp_list.append(cal_R_S(j))
        R_S.append(np.log(np.mean(tmp_list)))
        size.append(np.log(len(j)))

    results = np.polyfit(np.log(size), np.log(R_S), 1)

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
    for i in range(0, len(df)-n, n):
        h_df.loc[df[i:i+n].index[-1]] = hurst_exponent(range_, df[i:i+n])

    return h_df


if __name__ == '__main__':
    df = yf.download('^GSPC', '2019-01-01', '2021-9-30')
    data = cal_return(df.Close)
    range_ = [1, 3, 4, 6, 12]
    h_df = time_series_hurst_expo(64, data, range_)
    h_df_t = h_df.loc[h_df['hurst_exp'] > 0.54].index
    h_df_mr = h_df.loc[h_df['hurst_exp'] < 0.46].index
    plt.figure(figsize=(20, 10))
    df.plot()
    plt.scatter(h_df_t, df.loc[h_df_t], color='blue', marker='^', s=30)
    plt.scatter(h_df_mr, df.loc[h_df_mr], color='red', marker='v', s=30)
    plt.show()
