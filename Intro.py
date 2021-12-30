import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

def hurst_exponent(time_series, max_lag=20):

    lags = range(2, max_lag)

    tau = [np.std(np.array(time_series[lag:] - np.array(time_series[:-lag]))) for lag in lags]

    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

data = fdr.DataReader('US500')['Close']
h_df = pd.DataFrame(columns = ['hurst_exp'])
for i in range(0, len(data)-252):
    df = data.iloc[i:i+252]
    h_df.loc[df.index[-1], 'hurst_exp'] = hurst_exponent(df, 128)

h_df_t = h_df.loc[h_df['hurst_exp']>0.55].index
h_df_mr = h_df.loc[h_df['hurst_exp']<0.25].index

plt.figure(figsize=(40, 20))
plt.plot(data)
plt.scatter(h_df_t, data.loc[h_df_t], color='blue', s=10)
plt.scatter(h_df_mr, data.loc[h_df_mr], color='red', s=10)
plt.show()