from model.NuclearNormMinimizationModel import NuclearNormMinimizationModel
from model.OffensiveRatingSource import OffensiveRatingSource
from model.PaceSource import PaceSource
from model.FreeThrowsSource import FreeThrowsSource

from sklearn.metrics import mean_squared_error

import pickle
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import plotly as px

import sys

TEAMS = [
    "ATL",
    "BOS",
    "BRK",
    "CHO",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "HOU",
    "DET",
    "GSW",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHO",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]

URL = [
    "http://www.basketball-reference.com/leagues/NBA_2019_games-october.html",
    "http://www.basketball-reference.com/leagues/NBA_2019_games-november.html",
    "http://www.basketball-reference.com/leagues/NBA_2019_games-december.html",
]

def get_actual():
    # get box urls
    box_urls = []
    for url in URL:
        print("****", url)
        response = urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        soup.find_all("a")
        for link in soup.find_all("a"):
            if link.get("href").startswith("/boxscores/2"):
                box_urls.append(str(link.get("href")))
    pickle.dump(box_urls, open("box_urls.p", "wb"))

    # update data
    for url in box_urls:
        url = "http://www.basketball-reference.com" + url
        print(url)
        response = urlopen(url)
        html = response.read()
        stat_html = str(html).replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(stat_html, "html.parser")
        line_score_table = soup.find("table", id="line_score")
        stats = pd.read_html(str(line_score_table))[0]
        stats.columns = stats.columns.droplevel()
        stats = stats.rename({'Unnamed: 0_level_1': 'team'}, axis=1)
        a = stats.loc[0]["team"]
        b = stats.loc[1]["team"]
        score_a = stats.loc[0]["T"]
        score_b = stats.loc[1]["T"]
        truth.loc[a][b] = score_a
        truth.loc[b][a] = score_b
    truth.to_csv("cache/truth.csv")

def benchmark(predictions, update = False):
    truth = pd.DataFrame(columns = TEAMS, index = TEAMS)
    if update:
        get_actual()
    else:
        truth = pd.read_csv("cache/truth.csv")
        truth = truth.set_index("Unnamed: 0")

    mse = ((predictions - truth) ** 2).mean()
    mse.columns = ['team', 'mse']
    df = pd.DataFrame({'team': mse.index ,'mse' : mse.values})

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
    source = OffensiveRatingSource()
    data = source.get_data(urls)

    # solves the matrix
    model = NuclearNormMinimizationModel()
    predictions = model.predict(data)

    # cross validation
    df = benchmark(predictions)
    print("MSE: ", df)


    pd.options.plotting.backend = 'plotly'
    fig = df.plot.scatter(x='team', y='mse')
    fig.show()
    
    # write to predictions.csv
    predictions.to_csv("cache/predictions.csv")

    # get predictions
    if not args:
        matchups = [(a,b) for a in TEAMS for b in TEAMS if a is not b]
        for (a,b) in matchups:
            print(a,b)
            print(model.get_scores(a, b))
            print(val.loc[a][b])
    elif len(args) == 2:
        a = args[0].upper()
        b = args[1].upper()
        print(a,b)
        print(model.get_scores(a,b)) 
