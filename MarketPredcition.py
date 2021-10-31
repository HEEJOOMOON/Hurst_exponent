import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import yfinance as yf
import seaborn as sns
from scipy.stats import entropy

from DFA import *
from DMA import *

def hurst_dist(hurst_: pd.DataFrame,
               ) -> np.ndarray:
    '''

    :param hurst_: (pd.DataFrame) hurst_exponent according to time and scale
    :return: (np.ndarray) probability distribution function
    '''

    bins = [range(int(((max(hurst_)-min(hurst_)/0.008))))]

    return np.histogram(hurst_, bins=bins)


def auto_mutual_info():
    return None


def false_nearest():
    return None


def time_delay_vector():

    return None


def prediction():
    return None


if __name__=='__main__':
    data = yf.download('KO')
