# =============================================================================
# 1 - Packages
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# 2 - Feature Impact
# =============================================================================

#-----------
# Function 1 - Simulation Experiment Analysis
import pandas as pd

def generate_average_metrics_separated(lst_result_all):
    """
    Generates four DataFrames with the mean values of:
    - Cosine similarity metrics
    - Spearman correlation metrics
    - MAE metrics
    - MSE metrics

    Parameters
    ----------
    lst_result_all : list of lists of tuples
        Each inner list corresponds to an experiment and contains (key, value) metric pairs.

    Returns
    -------
    df_cosine_mean : pd.DataFrame
        Average of cosine similarity metrics.

    df_spearman_mean : pd.DataFrame
        Average of Spearman correlation metrics.

    df_mae_mean : pd.DataFrame
        Average of MAE metrics.

    df_mse_mean : pd.DataFrame
        Average of MSE metrics.
    """
    # Convert experiment results to DataFrame
    df = pd.DataFrame([{k: v for k, v in res} for res in lst_result_all])

    # Column groups
    cosine_cols = [col for col in df.columns if 'cossine' in col]
    spearman_cols = [col for col in df.columns if 'spearman' in col]
    mae_cols = [col for col in df.columns if 'mae' in col]
    mse_cols = [col for col in df.columns if 'mse' in col]

    # Compute means
    df_cosine_mean = df[cosine_cols].mean().to_frame(name='mean').reset_index().rename(columns={'index': 'metric'})
    df_spearman_mean = df[spearman_cols].mean().to_frame(name='mean').reset_index().rename(columns={'index': 'metric'})
    df_mae_mean = df[mae_cols].mean().to_frame(name='mean').reset_index().rename(columns={'index': 'metric'})
    df_mse_mean = df[mse_cols].mean().to_frame(name='mean').reset_index().rename(columns={'index': 'metric'})

    return df, df_cosine_mean, df_spearman_mean, df_mae_mean, df_mse_mean


# =============================================================================
# =============================================================================

