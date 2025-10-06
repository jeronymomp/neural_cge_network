# =============================================================================
# 1 - Packages
# =============================================================================
import random
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

#-----
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
    
#-----    
    def _generate_coef(self, normalize=False):
        rng = np.random.default_rng(self.random_state)
        coef = rng.uniform(0.1, 1.0, size=self.number_of_features)
        if normalize:
            coef = coef / coef.sum()
        return coef
    
#-----    
    def generate_log_cobb_douglas_data(self, min_data, max_data, normalize, intercept_one=False):
        """
        Generates synthetic data based on the original Cobb-Douglas production function (non-log-linear form).
    
        Cobb-Douglas formula:
            y = A * Π x_i^α_i
    
        Parameters:
            min_data (float): Minimum value for feature generation.
            max_data (float): Maximum value for feature generation.
            normalize (bool): Whether to normalize the generated coefficients so their sum equals 1.
            intercept_one (bool): If True, the intercept A will be set to 1 for all samples. 
                                  Otherwise, it will be randomly sampled from [min_data, max_data].
    
        Returns:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features), with positive values.
            y (np.ndarray): Target vector of shape (n_samples, 1), representing Cobb-Douglas outputs.
            coef (np.ndarray): Array of shape (n_features,) containing the true α coefficients.
        """
        rng = np.random.default_rng(self.random_state)
    
        # Intercept term A
        if intercept_one:
            intercept = np.ones(self.sample_size)
        else:
            intercept = np.array([random.randint(min_data, max_data) for _ in range(self.sample_size)])
    
        # Feature matrix X - normal dist such as make_regression
        X = rng.normal(0, 1, size=(self.sample_size, self.number_of_features))
        X = np.abs(X)
    
        # Cobb-Douglas coefficients α_i
        coef = self._generate_coef(normalize)
    
        # Calculate y = A * Π x_i^α_i
        powered = X ** coef[np.newaxis, :]         # Raise each feature to its corresponding α
        prod_linhas = np.prod(powered, axis=1)     # Product across features (per sample)
        y = prod_linhas * intercept                # Multiply by intercept A
        y = np.log(y)
        # Format outputs
        y = y.reshape(-1, 1).astype(np.float32)
        X = X.astype(np.float32)
        coef = coef.astype(np.float32)
    
        return X, y, coef

# =============================================================================
# =============================================================================

