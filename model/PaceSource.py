import pickle
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from model.Source import Source

class PaceSource(Source):

    file_name = "pace.csv"

    def make_matrices(self, urls) -> pd.DataFrame:
        """
        Makes matrices of offesive rating and pace
        Each entry in the matrix is the value (offensive rating or pace)
            of team1 against team2 (rows and columns respectively) for
            all games considered in the model.
        """
        box_urls = self.get_box_urls(urls)
        for url in box_urls:
            url = "http://www.basketball-reference.com" + url
            self.data = self.full_update(url, self.data)

    def get_box_urls(self, urls):
        """
        Gets all URLs for box scores (basketball-reference.com)
            from current season.

        Returns:
            box_urls (list): list of box score URLs from basketball reference
        """
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
        return box_urls

    def full_update(self, url: str, data: pd.DataFrame):
        """
        Updates the pace and offensive rating matrices for a given game.

        Args:
            url (str): URL to box score (basketball-reference.com)
            df_pace (pd.DataFrame): Pace DataFrame to update

        Returns:
            df_pace (pd.DataFrame):
                updated pace DataFrame
        """
        print(url)
        response = urlopen(url)
        html = response.read()
        stat_html = str(html).replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(stat_html, "html.parser")
        four_factors_table = soup.find("table", id="four_factors")
        table = pd.read_html(str(four_factors_table))[0]
        table.columns = table.columns.droplevel()

        team1 = table.loc[0][0]
        team2 = table.loc[1][0]
        pace = table.loc[1][1]

        data = self.update_df(data, team1, team2, pace)
        data = self.update_df(data, team2, team1, pace)
        return data
