import pickle
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup

class OffensiveRatingSource:
    def __init__(self, urls):
        """
        Attributes:
            urls (list): list of basketball reference URLs of games
                to include in model this needs to be manually updated
            teams (list): list of team canonical abbreviations
            box_urls (list): list of URLs to box scores for games
                included in model
            predictions (pd.DataFrame): DataFrame of predicted score.
                Each entry is the predicted score that the team in the
                index will score against each team in the columns.
                To predict a game, two lookups are required, one for
                each team against the other.
        """
        self.urls = urls
        self.teams = [
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
        if self.urls:
            self.box_urls = self.get_box_urls()
            self.df_OR = pd.DataFrame(0, index=self.teams, columns=self.teams)
            self.df_OR = self.make_matrices()
            self.write_matrices_to_csv()
        else:
            self.df_OR = pd.read_csv("model/OR.csv")
            self.df_OR = self.df_OR.set_index("Unnamed: 0")

    def get_box_urls(self):
        """
        Gets all URLs for box scores (basketball-reference.com)
            from current season.

        Returns:
            box_urls (list): list of box score URLs from basketball reference
        """
        box_urls = []
        for url in self.urls:
            print("****", url)
            response = urlopen(url)
            html = response.read()
            soup = BeautifulSoup(html, "html.parser")
            soup.find_all("a")
            for link in soup.find_all("a"):
                if link.get("href").startswith("/boxscores/2"):
                    box_urls.append(str(link.get("href")))
        pickle.dump(box_urls, open("box_urls.p", "wb"))
        return box_urls

    def get_stats(self, url: str):
        """
        Extracts statistics from URL

        Args:
            url (str): basketball-reference.com box score

        Returns:
            stats (pd.DataFrame): DataFrame of statistics from game
        """
        print(url)
        response = urlopen(url)
        html = response.read()
        stat_html = str(html).replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(stat_html, "html.parser")
        four_factors_table = soup.find("table", id="four_factors")
        stats = pd.read_html(str(four_factors_table))[0]
        stats.columns = stats.columns.droplevel()
        return stats

    def update_df(self, df: pd.DataFrame, team1: str, team2: str, value: int) -> pd.DataFrame:
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
        old_value = df.loc[team2][team1]
        if old_value == 0:
            new_value = float(value)
        else:
            new_value = (float(old_value) + float(value)) / 2
        df.loc[team2][team1] = new_value
        return df

    def extract_data(self, table: pd.DataFrame):
        """
        Extracts pace and offensive rating data from basketball-
            reference tables

        Args:
            table (pd.DataFrame): table of statistics scraped from
                basketball-reference contains advanced stats for a given games.

        Returns:
            team1 (str): Abbreviation of team1
            team2 (str): Abbreviation of team2
            team1_OR (float): Offensive rating of team1 (points per
                100 posessions)
            team2_OR (float): Offensive rating of team2 (points per
                100 posessions)
        """
        team1 = table.loc[0][0]
        team2 = table.loc[1][0]
        team1_OR = table.loc[0]["ORtg"]
        team2_OR = table.loc[1]["ORtg"]
        return team1, team2, team1_OR, team2_OR

    def full_update(self, url: str, df_OR: pd.DataFrame):
        """
        Updates the pace and offensive rating matrices for a given game.

        Args:
            url (str): URL to box score (basketball-reference.com)
            df_pace (pd.DataFrame): pace DataFrame to update
            df_OR (pd.DataFrame): Offensive Rating DataFrame to update

        Returns:
            df_pace, df_OR (pd.DataFrame, pd.DataFrame):
                updated pace and Offensive rating DataFrames
        """

        table = self.get_stats(url)
        team1, team2, team1_OR, team2_OR = self.extract_data(table)
        df_OR = self.update_df(df_OR, team1, team2, team1_OR)
        df_OR = self.update_df(df_OR, team2, team1, team2_OR)
        return df_OR


    def make_matrices(self) -> pd.DataFrame:
        """
        Makes matrices of offesive rating and pace
        Each entry in the matrix is the value (offensive rating or pace)
            of team1 against team2 (rows and columns respectively) for
            all games considered in the model.
        """
        df_OR = self.df_OR
        for url in self.box_urls:
            url = "http://www.basketball-reference.com" + url
            df_OR = self.full_update(url, df_OR)
        return df_OR

    def write_matrices_to_csv(self):
        """
        Writes pace and offensive ratings csv files.
        """
        self.df_OR.to_csv("./model/OR.csv")

    def get_data(self):
        return self.df_OR
