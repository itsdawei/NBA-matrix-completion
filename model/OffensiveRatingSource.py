import pickle
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from model.Source import Source

class OffensiveRatingSource(Source):

    file_name = "offensive_ratings.csv"

    def make_matrices(self, urls):
        """
        Makes matrices of pace
        Each entry in the matrix is the value (pace)
            of team 1 against team 2 (rows and columns respectively)
            for all games considered in the model.
        """
        # get box urls
        box_urls = []
        for url in urls:
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
            self.data = self.full_update(url, self.data)

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

        team1 = table.loc[0][0]
        team2 = table.loc[1][0]
        team1_OR = table.loc[0]["ORtg"]
        team2_OR = table.loc[1]["ORtg"]

        df_OR = self.update_df(df_OR, team1, team2, team1_OR)
        df_OR = self.update_df(df_OR, team2, team1, team2_OR)
        return df_OR
