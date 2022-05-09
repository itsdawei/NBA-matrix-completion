import pandas as pd

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
        return self.data

    def make_matrices(self, urls):
        """
        This method should modify self.data to the data matrix.
        Override this method for any implementation of Source.
        """
        pass
