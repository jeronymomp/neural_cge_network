# =============================================================================
# 1 - Packages
# =============================================================================
# External Packages
import time
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Built-in
from src.functions.data_generator import DGPGenerator
from src.functions.feature_impact import compute_linear_regression_feature_impact_percent,topk_pairwise_rank_agreement,topk_rank_correlation
from src.functions.feature_impact import cobb_douglas_impact,cobb_douglas_log_impact,model_alphas,compute_shap_impact_percent
from src.functions.feature_impact import compare_importance_cosine,normalize_importance,compare_importance_spearman
from src.functions.neural_network_function import FlexibleNeuralNetwork
from src.functions.cobb_douglas_act_function import CobbDouglasActivationFunction
from src.functions.log_cobb_douglas_act_function import CobbDouglasLogActivationFunction
from src.functions.cobb_doulgas_neural_network_function import CobbDouglasNeuralNetwork
from src.functions.log_cobb_doulgas_neural_network_function import LogCobbDouglasNeuralNetwork
from src.functions.result_analysis import generate_average_metrics_with_ci

# ==========================================================
# CONFIG
# ==========================================================
result_list = []
# IMPORTANT - Simulation experiment number
experiment_number = 1
n_sim = 2
#--------
# 2.1 - Experiment Parameters
sample_size = 10000
random_state = experiment_number
test_sample_size = 0.1
valid_sample_size = 0.1
sim_epochs = 500

#--------
# NAS Parameters
epochs_nas = 500
n_samples = 10

#--------
# Neural Network Parameters
epochs = 5000
neural_network_hidden_layers = [4,3]
learning_rate = 0.00001
tries_before_stopping = 5

#--------
# Cobb-Douglas Neural Network Parameters
sum_equal_one = 1


# Cossine table
cossine_tb = []
shock_tables = []

#--------
# Device - GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--------
# Pytorch Seed
random.seed(random_state)                          # Python
np.random.seed(random_state)                       # NumPy
torch.manual_seed(random_state)                    # PyTorch CPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)          # PyTorch GPU 
    torch.cuda.manual_seed_all(random_state)      # Multiple GPUs
torch.backends.cudnn.deterministic = True  # Deterministic
torch.backends.cudnn.benchmark = False     # Disable Non Deterministic


# ==========================================================
# SHOCK FUNCTIONS
# ==========================================================

def shock_effect_model(model,X,i,delta=0.1):

    X_up = X.copy()
    X_up[:, i] *= (1 + delta)

    y0 = model.predict(X)
    y1 = model.predict(X_up)

    return np.mean(y1-y0)


def shock_effect_nn(model,X_tensor,i,delta=0.1):

    X_up = X_tensor.clone()
    X_up[:,i] += delta

    with torch.no_grad():

        y0 = model(X_tensor).cpu().numpy()
        y1 = model(X_up).cpu().numpy()

    return np.mean(y1-y0)


def compute_y_log(X_log):
    return np.log(compute_y(np.exp(X_log)))

# =============================================================================
# Elasticity Real
# =============================================================================
def true_percent_impact(X, compute_y_fn, epsilon=0.01):
    """
    Compute true elasticity via % perturbation (model-agnostic)

    Parameters
    ----------
    X : np.ndarray
    compute_y_fn : function that maps X -> y
    epsilon : float (default 1%)

    Returns
    -------
    elasticity : np.ndarray (n_features,)
    """
    y_base = compute_y_fn(X)

    impacts = []

    for j in range(X.shape[1]):
        X_perturbed = X.copy()
        X_perturbed[:, j] *= (1 + epsilon)

        y_new = compute_y_fn(X_perturbed)

        impact = (y_new - y_base) / y_base
        impacts.append(np.mean(impact) / epsilon)

    return np.array(impacts)


# ==========================================================
# SHOCK - COBB DOUGLAS NN
# ==========================================================
def shock_effect_cobb(model, X_tensor, i, delta=0.1):

    X_up = X_tensor.clone()
    X_up[:, i] += delta

    with torch.no_grad():
        y0 = model(X_tensor).cpu().numpy().flatten()
        y1 = model(X_up).cpu().numpy().flatten()

    return np.mean(y1 - y0)


# ==========================================================
# COSINE SAFE
# ==========================================================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==========================================================
# RANDOM PRODUCTION FUNCTION
# ==========================================================
def generate_production(alpha, mode):

    def cobb(X):
        return np.prod(X ** alpha, axis=1)

    def log_cd(X):
        return np.exp(np.sum(alpha * np.log(X), axis=1))

    def sqrt_cd(X):
        return np.prod(np.sqrt(X) ** alpha, axis=1)

    def interaction(X):
        base = np.prod(X ** alpha, axis=1)
        inter = (X[:, 0] * X[:, 1]) ** 0.3
        return base * inter

    def ces_like(X):
        rho = 0.5
        return (np.sum(alpha * (X ** rho), axis=1)) ** (1 / rho)

    if mode == "cobb":
        return cobb
    elif mode == "log":
        return log_cd
    elif mode == "sqrt":
        return sqrt_cd
    elif mode == "interaction":
        return interaction
    else:
        return ces_like


def true_elasticity_log(X_log, compute_y_fn, eps=1e-4):
    base = compute_y_fn(X_log)

    elasticities = []

    for j in range(X_log.shape[1]):
        X_up = X_log.copy()
        X_up[:, j] += eps

        y_up = compute_y_fn(X_up)

        grad = (y_up - base) / eps
        elasticities.append(np.mean(grad))

    return np.array(elasticities)

# ==========================================================
# MONTE CARLO LOOP
# ==========================================================
for sim in range(n_sim):
    # Density of Leotief
    density = random.choice([0.3, 0.5, 0.7, 1.0])
    n = number_of_features = random.choice([3, 4, 5, 6, 8])
    chain_depth = random.choice([2, 3, 4, 5])

    # ----------------------------
    # DGP
    # ----------------------------
    mask = (np.random.rand(n, n) < density).astype(float)

    A = np.random.rand(n, n) * mask + 1e-8
    A = A / A.sum(axis=1, keepdims=True)
    A = A * 0.6

    L = np.linalg.inv(np.eye(n) - A)

    X = np.random.rand(sample_size, n) * 10 + 1

    alpha = L / L.sum(axis=1, keepdims=True)

    XL = X @ L.T

    Y = np.zeros((sample_size, n))
    for i in range(n):
        Y[:, i] = np.prod(XL ** alpha[i], axis=1)

    weights = np.random.rand(n)
    weights /= weights.sum()

    y = Y @ weights + np.random.randn(sample_size) * 0.5
    y = y.reshape(-1, 1)

    # ----------------------------
    # SPLIT
    # ----------------------------
    y = np.log(y)
    X = np.log(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sample_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_sample_size)
    
    X_test_raw = np.exp(X_test)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    # ----------------------------
    # Production CHAIN
    # ----------------------------
    chain_structure = []
    
    for d in range(chain_depth):
    
        layer_config = []
    
        for i in range(n):
    
            alpha_local = np.random.rand(n)
            alpha_local /= alpha_local.sum()
    
            mode = random.choice(["cobb", "log", "sqrt", "interaction", "ces"])
    
            layer_config.append({
                "alpha": alpha_local,
                "mode": mode
            })
    
        chain_structure.append(layer_config)
    
    # --------------------------------------------------
    # 🔴 FUNÇÃO ESTRUTURAL FIXA (AGORA SIM NÃO LINEAR)
    # --------------------------------------------------
    def compute_y(X_input):
    
        Z = X_input.copy()
    
        for d in range(chain_depth):
    
            Z = Z @ L.T  # input-output propagation
    
            Z_new = np.zeros_like(Z)
    
            for i in range(n):
    
                config = chain_structure[d][i]
    
                prod_fn = generate_production(
                    config["alpha"],
                    config["mode"]
                )
    
                Z_new[:, i] = prod_fn(Z)
    
            Z = Z_new
    
        return Z.mean(axis=1)
    
    elasticity_true = true_elasticity_log(X_test, compute_y_log)
    # ----------------------------
    # TRUE SHOCK
    # ----------------------------
    true_shocks = np.array([
        np.mean(compute_y(X_test + np.eye(n)[i] * 0.1) - compute_y(X_test))
        for i in range(n)
    ])

    # ======================================================
    # BENCHMARK MODELS
    # ======================================================
    models_results = []

    models = {
        "OLS": LinearRegression(),
        "RIDGE": Ridge(alpha=1),
        "LASSO": Lasso(alpha=0.1),
        "DecisionTree": DecisionTreeRegressor(max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)
    }

    for name, model in models.items():
        try:
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_test)

            def model_predict(X_input):
                return model.predict(X_input)

            if hasattr(model, "coef_"):
                elasticity = model.coef_
            else:
                def model_predict_log(X_input):
                    return model.predict(X_input)
                
                elasticity = true_elasticity_log(X_test, model_predict_log)

            models_results.append({
                "model": name,
                "model_obj": model,
                "mse": float(mean_squared_error(y_test, y_pred)),
                "elasticity": elasticity,
                "elasticity_rmse": float(np.sqrt(np.mean((elasticity - elasticity_true) ** 2))),
                "shock_rmse": float(np.sqrt(np.mean((shock_pred - true_shocks) ** 2)))
            })

        except Exception as e:
            print(f"[WARNING] {name} failed: {e}")

    # ======================================================
    # COBB DOUGLAS NN
    # ======================================================
    # 6.1 Function to perform random search
    def generate_random_cobb_config(number_of_features):
        number_of_layers = random.randint(1, 4)
        max_neurons = min(32, 4 * number_of_features)
        hidden_layers = sorted(
            [random.randint(number_of_features, max_neurons) for _ in range(number_of_layers)],
            reverse=True
        )
        dropout = random.choice([0.2, 0.3, 0.4, 0.5])
        return {
            "neural_network_hidden_layers": hidden_layers,
            "dropout": dropout
        }
    
    # 6.2 Generate n samples
    random_configs = [generate_random_cobb_config(number_of_features) for _ in range(n_samples)]
    
    # 6.3 Apply Random search on Base NN Architecture
    result_random_search_cobb = []
    
    for random_param in random_configs:
        cobb_fnn = LogCobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)
    
        cobb_model, cobb_train_losses, cobb_val_losses, cobb_best_val_loss, cobb_test_loss, cobb_architecture, cobb_weights = cobb_fnn.train_model(
        hidden_layers=random_param['neural_network_hidden_layers'],
        dropout_rate=random_param['dropout'],
        use_batchnorm=False,
        early_stopping_patience=tries_before_stopping,
        epochs=epochs_nas,
        normalized=sum_equal_one,
        optimizer_kwargs={"lr": learning_rate}
        )
    
        cobb_model.to(device)
        predicted_cobb = cobb_model(X_test_tensor).detach().cpu().numpy()
        mse_cobb = mean_squared_error(y_test, predicted_cobb)
        result_random_search_cobb.append(mse_cobb)
    
    # 6.4 Best result
    best_cobb_architecture = result_random_search_cobb.index(min(result_random_search_cobb))
    
    # 6.5 Best Parameters
    cobb_hidden_layers = random_configs[best_cobb_architecture]['neural_network_hidden_layers']
    cobb_dropout_rt = random_configs[best_cobb_architecture]['dropout']
    
    # Cobb-Douglas Neural Network
    cobb_fnn = LogCobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)
    
    cobb_model, cobb_train_losses, cobb_val_losses, cobb_best_val_loss, cobb_test_loss, cobb_architecture, cobb_weights = cobb_fnn.train_model(
    hidden_layers=cobb_hidden_layers,
    dropout_rate=cobb_dropout_rt,
    use_batchnorm=False,
    early_stopping_patience=tries_before_stopping,
    epochs=epochs,
    normalized=sum_equal_one,
    optimizer_kwargs={"lr": learning_rate}
    )
    
    cobb_model.to(device)
    cobb_layers = model_alphas(cobb_weights)

    cobb_elasticity = np.array([
        cobb_douglas_log_impact(cobb_layers, i, normalize=True)
        for i in range(n)
    ])

    cobb_shock = np.array([
        shock_effect_cobb(cobb_model, X_test_tensor, i)
        for i in range(n)
    ])

    # ======================================================
    # COSINE COMPARISON
    # ======================================================
    cosine_results = {}

    for res in models_results:
        cosine_results[res["model"]] = cosine_similarity(res["elasticity"], elasticity_true)

    cosine_results["CobbDouglasNN"] = cosine_similarity(cobb_elasticity, elasticity_true)

    cosine_df = pd.Series(cosine_results).sort_values(ascending=False)

    cossine_tb.append(cosine_df)
    
    
    # ===============================
    # SAVE ELASTICITY RMSE (ROBUST)
    # ===============================
    for res in models_results:
        result_list.append({
            "sim": sim,
            "model": res["model"],
            "elasticity_rmse": res["elasticity_rmse"]
        })
    
    # Cobb também
    result_list.append({
        "sim": sim,
        "model": "CobbDouglasNN",
        "elasticity_rmse": float(np.sqrt(np.mean((cobb_elasticity - elasticity_true) ** 2)))
    })
        
    
# ==========================================================
# FINAL RESULTS + CONFIDENCE INTERVAL
# ==========================================================

# --------
# 1. COSINE TABLE
# --------
cosine_all = pd.concat(cossine_tb, axis=1).T  # cada linha = simulação

cosine_summary = pd.DataFrame({
    "mean": cosine_all.mean(),
    "std": cosine_all.std(),
    "ci_lower": cosine_all.mean() - 1.96 * cosine_all.std() / np.sqrt(len(cosine_all)),
    "ci_upper": cosine_all.mean() + 1.96 * cosine_all.std() / np.sqrt(len(cosine_all))
})

cosine_summary.index.name = "model"


# --------
# 2. ELASTICITY RMSE TABLE
# --------
elasticity_df = pd.DataFrame(result_list)

elasticity_summary = elasticity_df.groupby("model")["elasticity_rmse"].agg(["mean", "std"])

elasticity_summary["ci_lower"] = (
    elasticity_summary["mean"] - 1.96 * elasticity_summary["std"] / np.sqrt(n_sim)
)

elasticity_summary["ci_upper"] = (
    elasticity_summary["mean"] + 1.96 * elasticity_summary["std"] / np.sqrt(n_sim)
)

# --------
# 3. MERGE FINAL (TABELA DO PAPER)
# --------
final_results = cosine_summary.merge(
    elasticity_summary,
    left_index=True,
    right_index=True,
    suffixes=("_cosine", "_elasticity")
)

print("\n===== FINAL RESULTS =====")
print(final_results)
