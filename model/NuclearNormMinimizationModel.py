import pandas as pd
import numpy as np
from cvxpy import *
from model.model import Model

class NuclearNormMinimization(Model):

    def predict(self, A: pd.DataFrame, mask) -> pd.DataFrame:
        """
        Solve using a nuclear norm approach, using CVXPY.
        [ Candes and Recht, 2009 ]
        Parameters:
        -----------
        A : m x n array
            matrix we want to complete

        Returns:
        --------
        X: m x n array
            completed matrix
        """
        X = Variable(shape=A.shape, name="X")
        objective = Minimize(norm(X, "nuc"))
        constraints = [multiply(mask, X) == mask * A]

        problem = Problem(objective, constraints)
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
