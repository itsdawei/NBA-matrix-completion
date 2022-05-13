import pandas as pd
import numpy as np

class Source:
    data = pd.DataFrame()
    root_path = "cache/"
    file_name = "default_source.csv"
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

    def get_data(self, urls = []):
        self.data = pd.DataFrame(index=self.teams, columns=self.teams)
        if urls:
            self.make_matrices(urls)
            print(self.data)
            self.data.to_csv(self.root_path + self.file_name)
        else:
            self.data = pd.read_csv(self.root_path + self.file_name)
            self.data = self.data.set_index("Unnamed: 0")
        return self.data.fillna(0)

    def make_matrices(self, urls):
        """
        This method should modify self.data to the data matrix.
        Override this method for any implementation of Source.
        """
        pass

    def update_df(self, df: pd.DataFrame, a: str, b: str, value: float) -> pd.DataFrame:
        """
        Updates df to add value of team1 and team2.
        For example, you can update the pace dataframe to add a game's pace.csv

        Args:
            df (pd.DataFrame): DataFrame to update
            team1: team on x axis index to update
            team2: team on columns to update
            value: value to add to DataFrame

        Returns:
            df (pd.DataFrame): updated DataFrame
        """
        old_value = df.loc[a][b]
        if old_value is np.nan:
            new_value = float(value)
        else:
            new_value = (float(old_value) + float(value)) / 2
        df.loc[a][b] = new_value
        return df
