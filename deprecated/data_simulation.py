"""
Class to create simulated data.

The objective is to generate series of explanatory variables X and a target variable Y,
which can result from linear and non-linear combinations of X.

Methods:
    - feature_generation: generates explanatory variables with random uniform values.
    - target_generation: (to be implemented) generates the target variable Y based on X.
"""

# =============================================================================
# Packages
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# Main class
# =============================================================================

class SeriesGenerator:
    """
    Class to generate simulated data.
    """
#=============================================================================#
    def __init__(self, seed: int = 0):
        """
        Initialize the SeriesGenerator and set the random seed for reproducibility.

        Parameters
        ----------
        seed : int, optional
            Random seed for numpy's random number generator (default is 0).
        """
        np.random.seed(seed)
        self.data_frame = None

#=============================================================================#        
    @staticmethod
    def linear_combination(df: pd.DataFrame, weights=None) -> pd.Series:
        """
        Combine features using given weights or default weights (1, 2, 3,...).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing explanatory variables.
        weights : list or np.ndarray, optional
            Weights for each feature column. If None, defaults to [1, 2, 3, ...].

        Returns
        -------
        pd.Series
            Target variable as weighted sum of features.
        """
        if weights is None:
            weights = np.arange(1, len(df.columns) + 1)
        else:
            weights = np.array(weights)
            if len(weights) != len(df.columns):
                raise ValueError("Length of weights must match number of feature columns.")

        y = np.zeros(len(df))
        for w, col in zip(weights, df.columns):
            y += w * df[col]
        return y

#=============================================================================#
    def feature_generation(self, n_variables: int, initial_interval: float, final_interval: float, n_obs: int) -> pd.DataFrame:
        """
        Generate a DataFrame containing a specified number of explanatory variables
        with randomly sampled values within a given interval.

        Each column X1, X2, ..., Xn is filled with n_obs random values drawn from a 
        uniform distribution between initial_interval and final_interval.

        Parameters
        ----------
        n_variables : int
            Number of explanatory variables (columns) to create.
        initial_interval : float
            Lower bound of the interval for generating random values.
        final_interval : float
            Upper bound of the interval for generating random values.
        n_obs : int
            Number of observations (rows); number of random values per column.

        Returns
        -------
        pd.DataFrame
            DataFrame with n_variables columns named 'X1', 'X2', ..., 'Xn'.
            Each column contains uniformly distributed random values.

        Example
        -------
        >>> gen = SeriesGenerator()
        >>> df = gen.feature_generation(3, 0, 10, 100)
        >>> df.shape
        (100, 3)
        >>> df.columns.tolist()
        ['X1', 'X2', 'X3']
        """
        data = {}
        for i in range(n_variables):
            data[f'X{i+1}'] = np.random.uniform(low=initial_interval, high=final_interval, size=n_obs)

        self.data_frame = pd.DataFrame(data)
        return self.data_frame

#=============================================================================#
    def target_generation(self, combine_func, **kwargs):
        """
        Generate a target variable 'y' as a combination of explanatory variables in self.data_frame.

        Parameters
        ----------
        combine_func : callable
            Function that takes the DataFrame of explanatory variables and returns a Series or array
            representing the target variable.

        Returns
        -------
        pd.Series
            The generated target variable.

        Raises
        ------
        ValueError
            If explanatory variables have not been generated yet.
        """
        if self.data_frame is None:
            raise ValueError("Explanatory variables not generated yet. Call feature_generation first.")
        self.data_frame['y'] = combine_func(self.data_frame, **kwargs)
        return self.data_frame['y']

# =============================================================================
# =============================================================================
