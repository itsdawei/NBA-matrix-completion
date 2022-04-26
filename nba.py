from model.model import NBAModel
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

    update = False;
    if "-u" in opts:
        update = True
    model = NBAModel(update)

    if not args:
        matchups = [(a,b) for a in TEAMS for b in TEAMS if a is not b]
        for (a,b) in matchups:
            model.get_scores(a, b)
    elif len(args) == 2:
        model.get_scores(args[0].upper(), args[1].upper())
