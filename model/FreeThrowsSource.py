import pickle
from urllib.request import urlopen

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from model.Source import Source

class FreeThrowsSource(Source):

    file_name = "free_throws.csv"

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

    def full_update(self, url: str, df_FT: pd.DataFrame):
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

        print(url)
        response = urlopen(url)
        html = response.read()
        stat_html = str(html).replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(stat_html, "html.parser")
        four_factors_table = soup.find("table", id="four_factors")
        table = pd.read_html(str(four_factors_table))[0]
        table.columns = table.columns.droplevel()

        a = table.loc[0][0]
        a_FT = table.loc[0]["FT/FGA"]
        b = table.loc[1][0]
        b_FT = table.loc[1]["FT/FGA"]

        df_FT = self.update_df(df_FT, a, b, a_FT)
        df_FT = self.update_df(df_FT, b, a, b_FT)
        return df_FT
