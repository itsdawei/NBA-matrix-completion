from model.NNMwithMSE import NuclearNormMinimizationMSE
from model.OffensiveRatingSource import OffensiveRatingSource
from model.PaceSource import PaceSource

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TEAMS = [ "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "HOU", "DET", "GSW",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI",
    "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]
N = len(TEAMS)
K = 0.8 # we observe 80% of entries

OR = OffensiveRatingSource().get_data()
PACE = PaceSource().get_data()
combined = OR * PACE / 100

def benchmark(truth, predictions):
    truth_no_nan = truth.replace(0, np.nan)

    mse = ((predictions - truth_no_nan) ** 2 ** 0.5).mean()
    df = pd.DataFrame({'team': mse.index ,'mse' : mse.values})

    # a = truth.fillnan(0)
    # rel_error = np.linalg.norm(predictions - a) / np.linalg.norm(a)
    # print(rel_error)
    return df

if __name__ == "__main__":
    print("------For K =", K)
    np.random.seed(123)
    mask = np.zeros((N,N))
    mask[:int(K * N)] = 1
    np.apply_along_axis(np.random.shuffle, 0, mask)
    for i in range(0, len(mask)):
        mask[i][i] = 0

# normalize df_OF
# max_OR = OR.to_numpy().max()
# norm_df_OR = OR / max_OR

    rmse_OR = []
    rmse_PACE = []
    rmse_combined = []
    x = np.linspace(0,80,17)
    print(x)
    for mu in x:
        print("------For mu =", mu)
        model = NuclearNormMinimizationMSE()
# recovered_OR = model.predict(norm_df_OR, mask)
        recovered_OR = model.predict(OR, mask, mu)
        recovered_PACE = model.predict(PACE, mask, mu)

        OR_mse_mean =  benchmark(OR, recovered_OR).loc[:,"mse"].mean()
        PACE_mse_mean = benchmark(PACE, recovered_PACE).loc[:,"mse"].mean()

        predictions = recovered_OR * recovered_PACE / 100
        combined_mse_mean = benchmark(combined, predictions).loc[:,"mse"].mean()

        rmse_OR.append(OR_mse_mean)
        rmse_PACE.append(PACE_mse_mean)
        rmse_combined.append(combined_mse_mean)

    # Plot the RMSE with different lambda
    fig, ax = plt.subplots()
    ax.plot(x, rmse_OR, linewidth=2.0, marker=".", label='Offensive Ratings')
    ax.plot(x, rmse_PACE, linewidth=2.0, marker=".", label='Pace')
    ax.plot(x, rmse_combined, linewidth=2.0, marker=".", label='Score Potential')
    ax.set_title("Root Mean Squared Error with Various Proportion of Observations.")
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Root Mean Squared Error (RMSE)')

    plt.xticks(np.linspace(0.1,1,10))
    plt.legend()
    plt.show()
