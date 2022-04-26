import pandas as pd
import cvxpy
from model.model import Model

class NuclearNormMinimizationModel(Model):

    def solve(self, A: pd.DataFrame) -> pd.DataFrame:
        """
        Solve using a nuclear norm approach, using CVXPY.
        [ Candes and Recht, 2009 ]
        Parameters:
        -----------
        A : m x n array
            matrix we want to complete
        mu : float
            hyperparameter controlling tradeoff between nuclear norm and square loss
        Returns:
        --------
        X: m x n array
            completed matrix
        """
        mask = A.transform(lambda x: x > 0)

        X = cvxpy.Variable(shape=A.shape, name="X")
        objective = cvxpy.Minimize(cvxpy.norm(X, "nuc"))
        constraints = [cvxpy.multiply(mask, X) == A]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=cvxpy.SCS)

        predictions = pd.DataFrame(X.value, columns=self.teams)
        predictions = predictions.assign(**{"Unnamed: 0": self.teams}).set_index(
            "Unnamed: 0"
        )

        assert predictions is not None
        self.predictions = predictions

        return predictions

    def get_scores(self, team1: str, team2: str):
        """
        Returns predicted score of two teams playing against each other.
        Teams can be in any order since home team advantage is not considered.

        Args:
            team1 (str): team1 abbreviation
            team2 (str): team2 abbreviation

        Returns:
        """
        return (self.predictions.loc[team1][team2], self.predictions.loc[team2][team1])
