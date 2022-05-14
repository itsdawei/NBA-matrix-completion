from model.NuclearNormMinimizationModel import NuclearNormMinimization
from model.NNMwithMSE import NuclearNormMinimizationMSE
from model.OffensiveRatingSource import OffensiveRatingSource
from model.PaceSource import PaceSource

import pickle
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

URL = [
    "http://www.basketball-reference.com/leagues/NBA_2019_games-october.html",
    "http://www.basketball-reference.com/leagues/NBA_2019_games-november.html",
    "http://www.basketball-reference.com/leagues/NBA_2019_games-december.html",
    "https://www.basketball-reference.com/leagues/NBA_2019_games-january.html",
    "https://www.basketball-reference.com/leagues/NBA_2019_games-february.html",
    "https://www.basketball-reference.com/leagues/NBA_2019_games-march.html",
    "https://www.basketball-reference.com/leagues/NBA_2019_games-april.html",
    "https://www.basketball-reference.com/leagues/NBA_2019_games-may.html",
    "https://www.basketball-reference.com/leagues/NBA_2019_games-june.html",
    ]

TEAMS = [ "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "HOU", "DET", "GSW",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI",
    "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
    ]

N = len(TEAMS)

def benchmark(truth, predictions, mask):
    mask = np.array(mask, dtype=bool)
    truth_nan = truth.replace(0, np.nan)

    mse = ((predictions.mask(mask) - truth_nan.mask(mask)) ** 2 ** 0.5).mean().mean()
    return mse

if __name__ == "__main__":
    """
    Main driver of the program
    Args:
        "-u": fetch update is included
        <team 1> <team 2>: first team and second team in the matchup; if not included, show
        all matchups
    """
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if len(args) > 2:
        raise SystemExit(f"Usage: {sys.argv[0]} [-u] <team_1> <team_2>")

    urls = []
    if "-u" in opts:
        urls = URL;

    # load data
    OR = OffensiveRatingSource().get_data(urls)
    PACE = PaceSource().get_data(urls)
    SCORE = OR * PACE / 100

    # generate mask
    OR_error_naive = []
    PACE_error_naive = []
    SCORE_error_naive = []
    OR_error_relaxed = []
    PACE_error_relaxed = []
    SCORE_error_relaxed = []

    x = np.linspace(0.1,1,10)

    for K in x:
        print("------For K =", K)
        np.random.seed(123)
        mask = np.zeros((N,N))
        mask[:int(K * N)] = 1
        np.apply_along_axis(np.random.shuffle, 0, mask)
        for i in range(0, len(mask)):
            mask[i][i] = 0

        # normalize df_OF
        max_OR = OR.to_numpy().max()
        norm_df_OR = OR / max_OR

        # print("rank", np.linalg.matrix_rank(OR, tol=0.5))
        # print("rank of norm_df_OR", np.linalg.matrix_rank(norm_df_OR, tol=0.5))

        # solves the matrix
        naive_model = NuclearNormMinimization()
        naive_recovered_OR = naive_model.predict(norm_df_OR, mask)
        naive_recovered_PACE = naive_model.predict(PACE, mask)

        relaxed_model = NuclearNormMinimizationMSE()
        relaxed_recovered_OR = relaxed_model.predict(norm_df_OR, mask, 1)
        relaxed_recovered_PACE = relaxed_model.predict(PACE, mask, 1)

        naive_recovered_OR = naive_recovered_OR * max_OR
        relaxed_recovered_OR = relaxed_recovered_OR * max_OR

        naive_recovered_SCORE = naive_recovered_OR * naive_recovered_PACE / 100
        relaxed_recovered_SCORE = relaxed_recovered_OR * relaxed_recovered_PACE / 100

        OR_error_naive.append(benchmark(OR, naive_recovered_OR, mask))
        PACE_error_naive.append(benchmark(PACE, naive_recovered_PACE, mask))
        SCORE_error_naive.append(benchmark(SCORE, naive_recovered_SCORE, mask))

        OR_error_relaxed.append(benchmark(OR, relaxed_recovered_OR, mask))
        PACE_error_relaxed.append(benchmark(PACE, relaxed_recovered_PACE, mask))
        SCORE_error_relaxed.append(benchmark(SCORE, relaxed_recovered_SCORE, mask))

        # write to predictions.csv
        # predictions.to_csv("cache/predictions.csv")

    # plot diff
    # fig, ax = plt.subplots()
    # ax.scatter(x, np.subtract(OR_error_naive, OR_error_relaxed), linewidth=2.0, marker=".", label='Offensive Ratings')
    # ax.scatter(x, np.subtract(PACE_error_naive, PACE_error_relaxed), linewidth=2.0, marker=".", label='Pace')
    # ax.scatter(x, np.subtract(SCORE_error_naive, SCORE_error_relaxed), linewidth=2.0, marker=".", label='Score Potential')
    # ax.set_xlabel('Proportion of Observed Entries (K)')
    # ax.set_ylabel('Root Mean Squared Error Difference')

    # plt.xticks(np.linspace(0.1,1,10))
    # plt.legend()
    # plt.show()

    # plot naive
    fig, ax = plt.subplots()
    ax.plot(x, OR_error_naive, linewidth=2.0, marker=".", label='Offensive Ratings')
    ax.plot(x, PACE_error_naive, linewidth=2.0, marker=".", label='Pace')
    ax.plot(x, SCORE_error_naive, linewidth=2.0, marker=".", label='Score Potential')
    ax.set_xlabel('Proportion of Observed Entries (K)')
    ax.set_ylabel('Root Mean Squared Error Difference')

    plt.xticks(np.linspace(0.1,1,10))
    plt.legend()
    plt.show()

    # plot relaxed
    fig, ax = plt.subplots()
    ax.plot(x, OR_error_relaxed, linewidth=2.0, marker=".", label='Offensive Ratings')
    ax.plot(x, PACE_error_relaxed, linewidth=2.0, marker=".", label='Pace')
    ax.plot(x, SCORE_error_relaxed, linewidth=2.0, marker=".", label='Score Potential')
    ax.set_xlabel('Proportion of Observed Entries (K)')
    ax.set_ylabel('Root Mean Squared Error Difference')

    plt.xticks(np.linspace(0.1,1,10))
    plt.legend()
    plt.show()

    # show predictions
    if not args:
        matchups = [(a,b) for a in TEAMS for b in TEAMS if a is not b]
        for (a,b) in matchups:
            print(a,b)
            print(predictions.loc[a][b], " ", predictions.loc[b][a]) 
    elif len(args) == 2:
        a = args[0].upper()
        b = args[1].upper()
        print(a,b)
        print(naive_recovered_SCORE.loc[a][b], " ", naive_recovered_SCORE.loc[b][a]) 
        print(naive_recovered_SCORE.loc[a][b], " ", naive_recovered_SCORE.loc[b][a]) 

