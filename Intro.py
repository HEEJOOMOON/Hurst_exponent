import pandas as pd
import numpy as np
import FinanceDataReader as fdr

import sys
sys.path.append('/home/hjmoon/PycharmProjects/Stochastic_Process/')
from OU_Process import Ornstein_Uhlenbeck
from Brownian_Motion import Brownian

def hurst_exponent(time_series, max_lag=20):

    lags = range(2, max_lag)

    tau = [np.std(np.array(time_series[lag:] - np.array(time_series[:-lag]))) for lag in lags]

    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


simulation = pd.DataFrame(columns=['mean_revert', 'gbm', 'trend'])
gbm = Brownian(process='geometric', period='daily').simulation(mu=0.05, sigma=0.25, n=10000)
mr = Ornstein_Uhlenbeck(period='daily').simulation(mu=0.1, theta=0.3, sigma=0.25, n=10000)
trend = Brownian(process='arithmetic', period='daily').simulation(mu=0.1, sigma=0.25, n=10000)
simulation['mean_revert'] = mr
simulation['gbm'] = gbm
simulation['trend'] = trend
max_lags = [50, 100, 150, 200, 250, 300, 350, 500, 1000]
results = pd.DataFrame(index=['mr', 'gbm', 'trend'], columns=max_lags)
for i in max_lags:
    results.loc['gbm'][i] = hurst_exponent(simulation['gbm'], max_lag=i)
    results.loc['mr'][i] = hurst_exponent(simulation['mean_revert'], max_lag=i)
    results.loc['trend'][i] = hurst_exponent(simulation['trend'], max_lag=i)

data = fdr.DataReader('US500', '2010-01-01')['Close']
for i in max_lags:
    print('Hurst exponent with lag %d: ' %i, hurst_exponent(data, i))