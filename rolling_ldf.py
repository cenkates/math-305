import numpy as np
import pandas as pd
from ldf import local_dependence
import os

def rolling_ldf_matrix(log_returns, window=60):
    tickers = log_returns.columns
    n = len(tickers)
    T = len(log_returns)

    ldf_results = []

    for t in range(window, T):
        window_data = log_returns.iloc[t-window:t]

        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ldf_vals = local_dependence(
                    window_data.iloc[:, i],
                    window_data.iloc[:, j],
                    bandwidth=0.05
                )
                mat[i, j] = np.mean(ldf_vals)  # pencere ortalaması

        ldf_results.append(mat)

    return np.array(ldf_results)
