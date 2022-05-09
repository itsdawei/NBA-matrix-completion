import pandas as pd
import cvxpy
from model.model import Model

class NuclearNormMinimization(Model):

    def predict(self, A: pd.DataFrame) -> pd.DataFrame:
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
        A = A.fillna(0)
        mask = A.transform(lambda x: x > 0)

        # cvx optimization problem
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
