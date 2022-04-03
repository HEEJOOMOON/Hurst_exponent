import pandas as pd
import numpy as np
import FinanceDataReader as fdr

import sys
sys.path.append('/home/hjmoon/PycharmProjects/Stochastic_Process/')
from OU_Process import Ornstein_Uhlenbeck
from Brownian_Motion import Brownian

def hurst_exponent(time_series, max_lag=20):

    lags = range(2, max_lag)
    time_series = np.log(time_series)
    tau = [np.std(np.array(time_series[lag:] - np.array(time_series[:-lag]))) for lag in lags]

    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


simulation = pd.DataFrame(columns=['mean_revert', 'gbm', 'trend'])
gbm = Brownian.simulation(process='geometric', mu=0.05, sigma=0.1, n=1000)
mr = Ornstein_Uhlenbeck.simulation(mu=0.1, theta=3, sigma=0.01, n=1000)
trend = Brownian.simulation(process='arithmetic', mu=300, sigma=0.25, n=1000)

simulation['mean_revert'] = mr+100
simulation['gbm'] = gbm
simulation['trend'] = trend

max_lags = [5, 10, 20, 50, 100, 200, 500]
results = pd.DataFrame(columns=['key', max_lags])
for idx in range(1000):
    for i in max_lags:
        results.loc[idx*3][i] = hurst_exponent(simulation['gbm'], max_lag=i)
        results.loc[idx*3+1][i] = hurst_exponent(simulation['mean_revert'], max_lag=i)
        results.loc[idx*3+2][i] = hurst_exponent(simulation['trend'], max_lag=i)
    results.loc[idx*3]['key'] = 'gbm'
    results.loc[idx*3+1]['key'] = 'mean_revert'
    results.loc[idx*3+2]['key'] = 'trend'

out = results.groupby('key').mean()

data = fdr.DataReader('US500', '2010-01-01')['Close']
for i in max_lags:
    print('Hurst exponent with lag %d: ' %i, hurst_exponent(data, i))