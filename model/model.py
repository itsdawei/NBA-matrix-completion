import pandas as pd

class Model:

    teams = [
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

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Solve the matrix by filling in the entries
        Parameters:
        -----------
        A : m x n array
            matrix we want to complete
        Returns:
        --------
        X: m x n array
            completed matrix
        """
        pass
