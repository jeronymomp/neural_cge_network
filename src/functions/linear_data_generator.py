# =============================================================================
# 1 - Packages
# =============================================================================
import numpy as np
from sklearn.datasets import make_regression 
from sklearn.linear_model import LinearRegression

# =============================================================================
# 2 - Data Generation
# =============================================================================
class LinearDataGenerator:
    """
    Generates synthetic data for a linear regression task.

    Attributes:
        sample_size (int): Number of samples to generate.
        number_of_features (int): Number of input features.
        random_state (int): Seed for reproducibility.

    Methods:
        generate_linear_regression_data():
            Generates and returns the feature matrix X, target vector y,
            and the true coefficients used in the data generation.
    """

    def __init__(self, sample_size, number_of_features, random_state):
        self.sample_size = sample_size
        self.number_of_features = number_of_features
        self.random_state = random_state

    def generate_linear_regression_data(self):
        """
        Generates synthetic linear regression data using scikit-learn's make_regression.

        Returns:
            X (np.ndarray): Feature matrix of shape (sample_size, number_of_features).
            y (np.ndarray): Target vector of shape (sample_size, 1).
            coef (np.ndarray): True coefficients used to generate y from X.
        """
        X, y, coef = make_regression(
            n_samples=self.sample_size,
            n_features=self.number_of_features,
            noise=0.1,
            coef=True,
            random_state=self.random_state
        )

        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1, 1)

        return X, y, coef

# =============================================================================
# =============================================================================

