import pandas as pd
import numpy as np
from cvxpy import *
from model.model import Model

class NuclearNormMinimizationMSE(Model):

    def predict(self, A: pd.DataFrame, mask) -> pd.DataFrame:
        """
        Solve using a nuclear norm approach, using CVXPY.
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
        X = cvxpy.Variable(shape=A.shape, name="X")
        mu = 1.0

        objective = Minimize(mu * norm(X, "nuc") + sum_squares(multiply(mask, X - A)))

        problem = Problem(objective, [])
        problem.solve(solver=SCS)

        predictions = pd.DataFrame(X.value, columns=self.teams)
        predictions = predictions.assign(**{"Unnamed: 0": self.teams}).set_index(
            "Unnamed: 0"
        )

        assert predictions is not None
        self.predictions = predictions

        nuc = np.sum(np.linalg.svd(predictions, compute_uv=False))
        print("nuc", nuc)

        return predictions
