from model.NuclearNormMinimizationModel import NuclearNormMinimization
from model.NNMwithMSE import NuclearNormMinimizationMSE
from model.OffensiveRatingSource import OffensiveRatingSource
from model.PaceSource import PaceSource

from sklearn.model_selection import train_test_split

import pickle
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import plotly.express as px

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
K = 0.5

def benchmark(truth, predictions):
    mse = ((predictions - truth) ** 2 ** 0.5).mean()
    df = pd.DataFrame({'team': mse.index ,'mse' : mse.values})

    print(df.loc[:,"mse"].mean())

    # a = truth.fillnan(0)
    # rel_error = np.linalg.norm(predictions - a) / np.linalg.norm(a)
    # print(rel_error)

    return df

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

    # generate mask
    np.random.seed(123)
    mask = np.zeros((N,N))
    mask[:int(K * N)] = 1
    np.apply_along_axis(np.random.shuffle, 0, mask)
    for i in range(0, len(mask)):
        mask[i][i] = 0

    # normalize df_OF
    # print("Normalizing OR...")
    max_OR = OR.to_numpy().max()
    norm_df_OR = OR / max_OR

    # print("rank", np.linalg.matrix_rank(OR, tol=0.5))
    # print("rank of norm_df_OR", np.linalg.matrix_rank(norm_df_OR, tol=0.5))

    # solves the matrix
    model = NuclearNormMinimization()
    # model = NuclearNormMinimizationMSE()
    recovered_OR = model.predict(norm_df_OR, mask)
    # recovered_OR = model.predict(OR, mask)
    recovered_PACE = model.predict(PACE, mask)

    recovered_OR = recovered_OR * max_OR

    print("rank", np.linalg.matrix_rank(recovered_OR, tol=0.5))

    # write to predictions.csv
    predictions = recovered_OR * recovered_PACE / 100
    predictions.to_csv("cache/predictions.csv")

    # S = np.linalg.svd(norm_df_OF, compute_uv=False, full_matrices=False)
    # _S = np.linalg.svd(predictions_OF, compute_uv=False, full_matrices=False)

    # f = px.scatter(S)
    # f.show()
    truth = OR * PACE / 100
    df_mse = benchmark(truth, predictions)
    print(df_mse)

    # plot the RMSE
    pd.options.plotting.backend = 'plotly'
    fig = df_mse.plot(kind='bar',x='team', y='mse', color='team',
            labels={'team':'Team', 'mse':'MSE'})
    fig.show()

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
        print(predictions.loc[a][b], " ", predictions.loc[b][a]) 
