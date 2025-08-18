"""
Functions:
    compute_linear_regression_feature_impact_percent
    cobb_douglas_impact
    model_alphas
    compute_shap_impact_percent
    compare_importance_cosine
    normalize_importance
    compare_importance_spearman
    
@author: jeronymo
"""

# =============================================================================
# 1 - Packages
# =============================================================================
import shap
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 2 - Feature Impact
# =============================================================================

#-----------
# Function 1 - Linear Regression Percentual Feature Impact 
def compute_linear_regression_feature_impact_percent(X, model: LinearRegression) -> np.ndarray:
    """
    Computes the average percentage impact of each feature on the model's predictions.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values  # Garante que X seja ndarray

    n_features = X.shape[1]
    y_pred_original = model.predict(X)
    impact_vector = []

    for i in range(n_features):
        X_perturbed = X.copy()
        X_perturbed[:, i] *= 1.01  # Aumenta a feature i em 1%
        y_pred_perturbed = model.predict(X_perturbed)

        safe_denominator = np.where(np.abs(y_pred_original) < 1e-8, 1e-8, y_pred_original)
        impact_percent = ((y_pred_perturbed - y_pred_original) / safe_denominator) * 100

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

    background = X_train[:min(background_size, len(X_train))]
    test_samples = X_test[:min(test_sample_size, len(X_test))]

    device = next(model.parameters()).device  # pega a device do modelo

    background_tensor = torch.from_numpy(background).float().to(device)
    test_tensor = torch.from_numpy(test_samples).float().to(device)

    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer.shap_values(test_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_output = np.mean(np.abs(y_train))
    impact_percent = (mean_abs_shap / mean_output) * 100

    return impact_percent

#-----------
# Function 5 - Cosine Similarity
def compare_importance_cosine(a, b, skip_normalization=False):
    """
    Computes the cosine similarity between two importance vectors.

    This method evaluates how similar the direction and distribution of importance is
    between two models. The result ranges from 0 (completely dissimilar) to 1 (identical direction).

    Args:
        a (np.ndarray): 1D array of importance values from model A.
        b (np.ndarray): 1D array of importance values from model B.
        skip_normalization (bool): If True, do not normalize the importance vectors.

    Returns:
        float: Cosine similarity between the (optionally normalized) importance vectors.
    """
    if not skip_normalization:
        a = normalize_importance(a)
        b = normalize_importance(b)
    
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

#-----------
# Function 6 - Normalize Importance
def normalize_importance(array):
    """
    Normalizes an array of feature importance values so that they sum to 1.

    This is useful for comparing relative importance across features or models.
    The function takes the absolute value of each element to ignore sign.

    Args:
        array (np.ndarray): 1D array of importance values (can be positive or negative).

    Returns:
        np.ndarray: Normalized array of the same shape, with values summing to 1.
    """
    array = np.abs(array)
    total = array.sum()
    return array / total if total != 0 else array

#-----------
# Function 7 - Importance Spearmean
def compare_importance_spearman(a, b, skip_normalization=False):
    """
    Compares the relative importance rankings of two coefficient arrays using Spearman correlation.

    This method evaluates the monotonic relationship between the importances
    of two sets of features. A value of 1.0 indicates perfect agreement in ranking,
    -1.0 indicates reversed ranking, and 0.0 means no correlation.

    Args:
        a (np.ndarray): 1D array of importance values from model A.
        b (np.ndarray): 1D array of importance values from model B.
        skip_normalization (bool): If True, do not normalize the importance vectors.

    Returns:
        float: Spearman rank correlation coefficient between the (optionally normalized) vectors.
    """
    if not skip_normalization:
        a = normalize_importance(a)
        b = normalize_importance(b)
    
    corr, _ = spearmanr(a, b)
    return corr

#-----------
# Function 8 - Log Cobbb Douglas Impact
def cobb_douglas_log_impact(layers_exponents, feature_index, normalize=False):
    """
    Computes the cumulative influence of a specific input feature on the final output
    of a Cobb-Douglas neural network operating in log-space.

    In the log-space formulation, the network applies additive transformations instead 
    of multiplicative ones. The feature impact is propagated through each layer by matrix 
    multiplication, simulating the effect of a unit feature signal.

    Args:
        layers_exponents (list of np.ndarray): A list of exponent matrices (alphas) 
            for each layer of the network. Each matrix should have shape 
            (output_dim, input_dim).
        feature_index (int): The index of the input feature for which to compute the impact.
        normalize (bool): If True, the result is normalized by the total impact 
            across all input features, returning a fraction in [0, 1].

    Returns:
        float: The raw or normalized cumulative impact of the specified feature 
        on the final network output.
    """
    # Unit vector to simulate isolated influence of the selected feature
    impact_vector = np.zeros(layers_exponents[0].shape[1])
    impact_vector[feature_index] = 1.0

    # Forward propagate through all exponent (alpha) layers in log-space
    for layer_exp in layers_exponents:
        impact_vector = layer_exp @ impact_vector  # shape: (output_dim,)

    total_impact = impact_vector.sum()

    if normalize:
        total_sum = sum(
            cobb_douglas_log_impact(layers_exponents, i, normalize=False)
            for i in range(layers_exponents[0].shape[1])
        )
        total_impact = total_impact / total_sum if total_sum != 0 else 0.0

    return total_impact



#-----------------------------------------------------------------------------#

# =============================================================================
# =============================================================================

