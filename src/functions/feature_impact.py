# =============================================================================
# 1 - Packages
# =============================================================================
import numpy as np

# =============================================================================
# 2 - Feature Impact
# =============================================================================
# Function 1 - Linear Regression Percentual Feature Impact 
def compute_linear_regression_feature_impact_percent(X: np.ndarray, model: LinearRegression) -> np.ndarray:
    """
    Computes the average percentage impact of each feature on the model's predictions.
    
    The method perturbs each feature by +1% and measures the average relative change
    in the predicted output.

    Args:
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        model (LinearRegression): A trained linear regression model.

    Returns:
        np.ndarray: A 1D array with the mean percentage impact of each feature.
    """
    n_features = X.shape[1]
    y_pred_original = model.predict(X)
    impact_vector = []

    for i in range(n_features):
        X_perturbed = X.copy()
        X_perturbed[:, i] *= 1.01  # Increase feature by 1%
        y_pred_perturbed = model.predict(X_perturbed)
        impact_percent = ((y_pred_perturbed - y_pred_original) / y_pred_original) * 100
        mean_impact = np.mean(impact_percent)
        impact_vector.append(mean_impact)

    return np.array(impact_vector)

#-----------------------------------------------------------------------------#

# =============================================================================
# =============================================================================

