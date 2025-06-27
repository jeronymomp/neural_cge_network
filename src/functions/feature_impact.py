# =============================================================================
# 1 - Packages
# =============================================================================
import shap
import torch
import numpy as np
from sklearn.linear_model import LinearRegression

# =============================================================================
# 2 - Feature Impact
# =============================================================================

#-----------
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

#-----------
# Function 2  -  Cobbb Douglas Impact
def cobb_douglas_impact(layers_exponents, feature_index, normalize=False):
    """
    Computes the cumulative impact of a specific input feature on the final output
    of a Cobb-Douglas neural network.

    The function performs a forward propagation of a unit impact vector through the
    exponent matrices (alphas) of each layer. The final result is the sum of contributions,
    expressed as a percentage.

    Args:
        layers_exponents (list of np.ndarray): List of exponent matrices (alphas) for each layer,
                                               where each matrix is of shape (output_dim, input_dim).
        feature_index (int): Index of the input feature to compute the impact for.
        normalize (bool): If True, returns the normalized impact (between 0 and 1) relative
                          to the total impact of all features.

    Returns:
        float: The (normalized or raw) impact of the specified feature on the output,
               as a percentage if not normalized, or as a fraction if normalized.
    """
    impacto = np.zeros(layers_exponents[0].shape[1])
    impacto[feature_index] = 1.0
    for layer_exp in layers_exponents:
        impacto = layer_exp @ impacto  # Multiplicação de matriz
    impacto_final = impacto.sum() * 100  # impacto percentual

    if normalize:
        impacto_total = sum([
            cobb_douglas_impact(layers_exponents, i, normalize=False)
            for i in range(layers_exponents[0].shape[1])
        ])
        impacto_final = impacto_final / impacto_total if impacto_total != 0 else 0.0

    return impacto_final

#-----------
# Function 3 - Extraction of alphas
def model_alphas(model_state_dict):
    """
    Extracts all Cobb-Douglas exponent parameters ('alphas') from a model's state dictionary.

    It looks for all keys in the state dict that include the string "alphas",
    assuming they correspond to layers of type CobbDouglasActivationFunction.

    Args:
        model_state_dict (OrderedDict): The state dictionary from a PyTorch model,
                                        typically obtained via model.state_dict().

    Returns:
        list of np.ndarray: A list containing the exponent matrices (alphas) for each layer.
    """
    alphas = []
    for key in model_state_dict:
        if "alphas" in key:
            alphas.append(model_state_dict[key].cpu().numpy())
    return alphas

#-----------
# Function 4 - SHAP Values
def compute_shap_impact_percent(model, X_train, X_test, y_train,
                                       background_size=100, test_sample_size=50):
    """
    Computes the approximate percentage impact of each feature based on SHAP values.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model.
    X_train : np.ndarray
        Training data (input to the model).
    X_test : np.ndarray
        Test data (input to the model).
    y_train : np.ndarray
        Ground truth target values.
    background_size : int, optional
        Number of background samples to use for SHAP (default: 100).
    test_sample_size : int, optional
        Number of test samples to explain (default: 50).

    Returns:
    --------
    impact_percent : np.ndarray
        Approximate percentage impact of each feature on the model's output.
    """
    model.eval()

    # Select background and test samples (limited to data size)
    background = X_train[:min(background_size, len(X_train))]
    test_samples = X_test[:min(test_sample_size, len(X_test))]

    # Convert to PyTorch tensors
    background_tensor = torch.from_numpy(background).float()
    test_tensor = torch.from_numpy(test_samples).float()

    # Create SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, background_tensor)

    # Compute SHAP values
    shap_values = explainer.shap_values(test_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Compute mean of target (used to normalize)
    mean_output = np.mean(np.abs(y_train))

    # Compute percentage impact relative to the mean output
    impact_percent = (mean_abs_shap / mean_output) * 100

    return impact_percent

#-----------
# Function 5 - Cosine Similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # avoid division by zero
    return dot_product / (norm_vec1 * norm_vec2)

#-----------------------------------------------------------------------------#

# =============================================================================
# =============================================================================

