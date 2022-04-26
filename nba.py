from model.NuclearNormMinimizationModel import NuclearNormMinimizationModel
from model.OffensiveRatingSource import OffensiveRatingSource
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

    if "-u" not in opts:
        URL = [];

    # load data
    data = OffensiveRatingSource(URL)

    # solves the matrix
    model = NuclearNormMinimizationModel()
    model.solve(data.get_data())

    # get predictions
    if not args:
        matchups = [(a,b) for a in TEAMS for b in TEAMS if a is not b]
        for (a,b) in matchups:
            print(a,b)
            print(model.get_scores(a, b))
    elif len(args) == 2:
        a = args[0].upper()
        b = args[1].upper()
        print(a,b)
        print(model.get_scores(a,b)) 
