# =============================================================================
# Packages
# =============================================================================
import numpy as np

# =============================================================================
# Class for DGPs
# =============================================================================
class DGPGenerator:
    """Class to generate synthetic data from various Data Generating Processes (DGPs)
    and compute feature percent impacts.
    """

    # -------------------------------------------------------------------------
    # Beta generators
    # -------------------------------------------------------------------------
    @staticmethod
    def generate_betas_sum_1(p: int) -> np.ndarray:
        betas = np.random.rand(p)
        return betas / betas.sum()

    @staticmethod
    def generate_betas_positive(p: int, low: float = 0.1, high: float = 1.0) -> np.ndarray:
        return np.random.uniform(low, high, size=p)

    # -------------------------------------------------------------------------
    # DGPs
    # -------------------------------------------------------------------------
    @staticmethod
    def linear(n: int, p: int):
        X = np.random.uniform(0, 1, size=(n, p))
        beta = DGPGenerator.generate_betas_positive(p)
        epsilon = np.random.normal(0, 0.1, size=n)
        y = X @ beta + epsilon
        return X, y, beta

    @staticmethod
    def cobb_douglas(n: int, p: int, return_log: bool = False, eps: float = 1e-6):
        X = np.random.uniform(eps, 1, size=(n, p))
        alphas = DGPGenerator.generate_betas_sum_1(p)
        log_y = np.sum(np.log(X) * alphas, axis=1)
        y = np.exp(log_y)
        if return_log:
            return X, log_y, alphas
        else:
            return X, y, alphas

    @staticmethod
    def nonlinear_interactions(n: int, p: int = 2):
        if p < 2:
            raise ValueError("p must be at least 2 for interactions model")
        X = np.random.uniform(0, 1, size=(n, 2))
        X_inter = np.column_stack((X, X[:,0] * X[:,1]))
        betas = DGPGenerator.generate_betas_positive(3)
        epsilon = np.random.normal(0, 0.1, size=n)
        y = betas[0]*X[:,0] + betas[1]*X[:,1]**2 + betas[2]*X[:,0]*X[:,1] + epsilon
        return X_inter, y, betas

    @staticmethod
    def factor_model(n: int, p: int, k: int = 2):
        F = np.random.normal(0, 1, size=(n, k))
        Lambda = np.random.uniform(0.5, 1.5, size=(k, p))
        V = np.random.normal(0, 0.5, size=(n, p))
        X = np.abs(F @ Lambda + V)
        beta = DGPGenerator.generate_betas_positive(p)
        epsilon = np.random.normal(0, 0.1, size=n)
        y = X @ beta + epsilon
        return X, y, beta

    # -------------------------------------------------------------------------
    # Percent feature impact
    # -------------------------------------------------------------------------
    @staticmethod
    def percent_feature_impact(X: np.ndarray, y: np.ndarray, coefs: np.ndarray, 
                               model_type: str = 'linear', log_output: bool = False):
        """
        Compute the percent impact of each feature on the output.
        
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix (n_samples x n_features)
        y : np.ndarray
            Response vector (n_samples,)
        coefs : np.ndarray
            Coefficients of the model (beta or alpha)
        model_type : str
            Type of model: 'linear', 'factor', 'cobb_douglas', 'nonlinear'
        log_output : bool
            Only for Cobb-Douglas: whether y is log(y)
        
        Returns
        -------
        impacts : np.ndarray
            Percent impact of each feature
        """
        if model_type in ['linear', 'factor']:
            return 100 * (coefs * X.mean(axis=0)) / y.mean()
        
        elif model_type == 'cobb_douglas':
            if log_output:
                return coefs * 100  # elasticity in %
            else:
                log_X_mean = np.log(X).mean(axis=0)
                log_y_mean = np.sum(log_X_mean * coefs)
                return 100 * coefs * log_X_mean / log_y_mean
        
        elif model_type == 'nonlinear':
            x1, x2 = X[:,0], X[:,1]
            term1 = coefs[0]*x1
            term2 = coefs[1]*x2**2
            term3 = coefs[2]*x1*x2
            return 100 * np.array([term1.mean(), term2.mean(), term3.mean()]) / y.mean()
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
