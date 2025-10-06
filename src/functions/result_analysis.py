# =============================================================================
# 1 - Packages
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# 2 - Feature Impact
# =============================================================================
def generate_average_metrics_with_ci(lst_result_all, confidence=0.95):
    """
    Generates DataFrames with mean and confidence intervals for selected metrics.
    
    Parameters
    ----------
    lst_result_all : list of list of tuples
        Each element is a list of tuples (metric_name, value) for a single experiment.
    confidence : float, optional
        Confidence level for CI (default 0.95).
    
    Returns
    -------
    tuple of pd.DataFrame
        - df: raw metrics as DataFrame
        - df_cosine_ci, df_spearman_ci, df_mae_ci, df_mse_ci
          DataFrames with mean and CI for each metric group.
        - df_topk_pair_ci, df_topk_rank_ci
          DataFrames with mean and CI for top-k pairwise metrics.
    """
    
    # Convert list of tuples to DataFrame
    df = pd.DataFrame([{k: v for k, v in res} for res in lst_result_all])
    
    # Define metric groups
    cosine_cols = [col for col in df.columns if 'cossine' in col]
    spearman_cols = [col for col in df.columns if 'spearman' in col]
    mae_cols = [col for col in df.columns if 'mae' in col]
    mse_cols = [col for col in df.columns if 'mse' in col]
    
    # Top-k separate
    topk_pair_cols = [col for col in df.columns if 'topk_pair' in col]
    topk_rank_cols = [col for col in df.columns if 'topk_rank' in col]
    
    # Function to compute mean and confidence interval
    def compute_mean_ci(data):
        mean = np.mean(data)
        sem = stats.sem(data, nan_policy='omit')
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        return mean, mean - h, mean + h
    
    # Create DataFrame with CI for a group of metrics
    def create_ci_df(columns):
        results = []
        for col in columns:
            mean, ci_lower, ci_upper = compute_mean_ci(df[col].dropna())
            results.append({
                'metric': col,
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_95': f"{mean:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
            })
        return pd.DataFrame(results)
    
    # Create CI DataFrames
    df_cosine_ci = create_ci_df(cosine_cols)
    df_spearman_ci = create_ci_df(spearman_cols)
    df_mae_ci = create_ci_df(mae_cols)
    df_mse_ci = create_ci_df(mse_cols)
    df_topk_pair_ci = create_ci_df(topk_pair_cols)
    df_topk_rank_ci = create_ci_df(topk_rank_cols)
    
    return df, df_cosine_ci, df_spearman_ci, df_mae_ci, df_mse_ci, df_topk_pair_ci, df_topk_rank_ci

# =============================================================================
# =============================================================================
