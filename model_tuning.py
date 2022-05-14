from model.NNMwithMSE import NuclearNormMinimizationMSE
from model.OffensiveRatingSource import OffensiveRatingSource
from model.PaceSource import PaceSource

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

TEAMS = [ "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "HOU", "DET", "GSW",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI",
    "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]
N = len(TEAMS)
K = 0.9 # we observe 80% of entries

OR = OffensiveRatingSource().get_data()
PACE = PaceSource().get_data()
combined = OR * PACE / 100

def benchmark(truth, predictions, mask):
    mask = np.array(mask, dtype=bool)
    truth_nan = truth.replace(0, np.nan)

    mse = ((predictions.mask(mask) - truth_nan.mask(mask)) ** 2 ** 0.5).mean().mean()
    return mse

if __name__ == "__main__":
    print("------For K =", K)
    np.random.seed(123)
    mask = np.zeros((N,N))
    mask[:int(K * N)] = 1
    np.apply_along_axis(np.random.shuffle, 0, mask)
    for i in range(0, len(mask)):
        mask[i][i] = 0

    MU = Parameter(nonneg=True)
    X = Variable(shape=OR.shape, name="X")
    Y = Variable(shape=PACE.shape, name="Y")

    problemOR = Problem(Minimize(multiply(MU, norm(X, "nuc")) + 0.5 * sum_squares(multiply(mask, X - OR))), [])
    problemPACE = Problem(Minimize(multiply(MU, norm(Y, "nuc")) + 0.5 * sum_squares(multiply(mask, Y - PACE))), [])

    rmse_OR = []
    rmse_PACE = []
    rmse_combined = []
    x = np.linspace(0,40,9)
    for mu in x:
        print("------For mu =", mu)
        MU.value = mu

        problemOR.solve(solver=SCS)
        recovered_OR = pd.DataFrame(X.value, columns=TEAMS)
        recovered_OR = recovered_OR.assign(**{"Unnamed: 0": TEAMS}).set_index(
            "Unnamed: 0"
        )

        problemPACE.solve(solver=SCS)
        recovered_PACE = pd.DataFrame(Y.value, columns=TEAMS)
        recovered_PACE = recovered_PACE.assign(**{"Unnamed: 0": TEAMS}).set_index(
            "Unnamed: 0"
        )

        OR_mse_mean = benchmark(OR, recovered_OR, mask)
        PACE_mse_mean = benchmark(PACE, recovered_PACE, mask)

        predictions = recovered_OR * recovered_PACE / 100
        combined_mse_mean = benchmark(combined, predictions, mask)

        rmse_OR.append(OR_mse_mean)
        rmse_PACE.append(PACE_mse_mean)
        rmse_combined.append(combined_mse_mean)

    print(rmse_OR)
    print(rmse_PACE)
    print(rmse_combined)

    # Plot the RMSE with different lambda
    fig, ax = plt.subplots()
    ax.plot(x, rmse_OR, linewidth=2.0, marker=".", label='Offensive Ratings')
    ax.plot(x, rmse_PACE, linewidth=2.0, marker=".", label='Pace')
    ax.plot(x, rmse_combined, linewidth=2.0, marker=".", label='Score Potential')
    ax.set_title("Root Mean Squared Error for Various Lambda")
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Root Mean Squared Error (RMSE)')

    print(x)
    plt.xticks(x)
    plt.legend()
    plt.show()
