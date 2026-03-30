#--------
# Working Directory
import os
wkdir = "/home/jeronymo/Downloads/neural_cge_network"
os.chdir(wkdir)

# =============================================================================
# 1 - Packages
# =============================================================================
# External Packages
import time
import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.metrics import mean_absolute_error, mean_squared_error
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

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

# =============================================================================
# 2 - Parameters
# =============================================================================
result_list = []
# IMPORTANT - Simulation experiment number
experiment_number = 11

#--------
# 2.1 - Experiment Parameters
sample_size = 10000
number_of_features = 5
random_state = experiment_number
test_sample_size = 0.2
valid_sample_size = 0.2
sim_epochs = 1000

#--------
# NAS Parameters
epochs_nas = 500
n_samples = 10

#--------
# Neural Network Parameters
epochs = 1000
neural_network_hidden_layers = [4,3]
learning_rate = 0.00001
tries_before_stopping = 5

#--------
# Cobb-Douglas Neural Network Parameters
sum_equal_one = 1

# Time series Genration
initial_date = '1973-01-01'

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

# =============================================================================
# Functions
# =============================================================================
# -----------------------------------------------------------------------------
# 1. Stationary AR generator
# -----------------------------------------------------------------------------
def generate_stationary_ar(p):
    """
    Generates stationary AR(p) coefficients.
    """
    while True:
        coefs = np.random.uniform(-0.9, 0.9, size=p)
        ar = np.r_[1, -coefs]
        ma = np.array([1])

        process = ArmaProcess(ar, ma)

        if process.isstationary:
            return coefs


# -----------------------------------------------------------------------------
# 2. Breakthru function (corrected)
# -----------------------------------------------------------------------------
def breakthru(dgp_list, dgp_sample_size, initial_date, burnin=200):
    """
    Simulates a time series with possible structural breaks 
    (multiple ARMA regimes).
    """

    if isinstance(dgp_sample_size, int):
        dgp_sample_size = [dgp_sample_size]

    simulated_series = []

    for i in range(len(dgp_list)):
        ar_coefs, ma_coefs = dgp_list[i]

        ar = np.r_[1, -np.array(ar_coefs)]
        ma = np.r_[1, np.array(ma_coefs)]

        process = ArmaProcess(ar, ma)

        if not process.isstationary:
            raise ValueError(f"AR process in regime {i} is not stationary")

        data = process.generate_sample(
            nsample=dgp_sample_size[i] + burnin
        )

        data = data[burnin:]
        simulated_series.extend(data)

    total_n = sum(dgp_sample_size)

    time_index = pd.date_range(
        start=initial_date,
        periods=total_n,
        freq='ME'
    )

    ts_series = pd.DataFrame({
        'sim_data': simulated_series,
        'time': time_index
    })

    return ts_series


# -----------------------------------------------------------------------------
# 3. Time-series feature generator (X)
# -----------------------------------------------------------------------------
def generate_X_time_series(number_of_features, sample_size, initial_date):
    X_list = []

    for _ in range(number_of_features):
        number_of_lags = random.choice([1, 2, 3])

        # stationary AR
        coefs = generate_stationary_ar(number_of_lags)

        # valid MA (avoids non-invertibility)
        ma = []

        dgp_list = [[coefs, ma]]
        dgp_sample_size = [sample_size]

        df = breakthru(dgp_list, dgp_sample_size, initial_date)

        X_list.append(df['sim_data'].values)

    # shape (n, p)
    X = np.column_stack(X_list)

    return X


# -----------------------------------------------------------------------------
# 4. Final dataset generator
# -----------------------------------------------------------------------------
def generate_dataset_ts(n, p, initial_date):
    """
    Generates a dataset with time-dependent features X and linear target y.
    """

    X = generate_X_time_series(p, n, initial_date)

    beta = np.random.uniform(0.1, 1.0, size=p)

    epsilon = np.random.normal(0, 0.1, size=n)

    y = X @ beta + epsilon

    return X, y, beta

# =============================================================================
# Starting Loop
# =============================================================================
for i in range(0,sim_epochs):
    # Start Time
    start_time = time.time()
    # Generate random sample
    random_state = random.randint(0,10000000)
    # Generate Random Number of Features
    number_of_features = random.randint(1,5)
    # Generate Random Number of layers
    number_of_layers = random.randint(1,5)
    # Generate Random Number of Neurons
    max_neurons = min(32, 4 * number_of_features)
    neural_network_hidden_layers = sorted(
        [random.randint(number_of_features, max_neurons) for _ in range(number_of_layers)],
        reverse=True)
    # Generate Random Dropout
    dropout = random.choice([0.2, 0.3, 0.4, 0.5])
    # Generate Random Act Function for Benchmark
    activ = [
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.SELU,
    nn.Softplus]
    
    activition_function = random.choice(activ)

    # =============================================================================
    # 3 - Data Generation and Preparation
    # =============================================================================
    X, y, beta = generate_dataset_ts(
        n=sample_size,
        p=number_of_features,
        initial_date='2000-01-01'
    )
    y = y.reshape(-1, 1)
    
    var_x = X.var(axis=0)
    raw_impact = (beta ** 2) * var_x
    real_feature_impact = raw_impact / raw_impact.sum()
    
    # Train Test Split (with numpy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sample_size, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_sample_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    # Random Assigment
    min_val = X_train.min()
    max_val = X_train.max()
    random_feature_impact = np.random.uniform(low=min_val, high=max_val, size=real_feature_impact.shape)

    # Convert splits to tensors on device (GPU/CPU)
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_valid_tensor = torch.from_numpy(X_valid).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    # =============================================================================
    # 4 - Neural Networks Random Config Params and Random Search for Base NN
    # =============================================================================
    # 4.1 Function to perform random search
    def generate_random_config(number_of_features):
        number_of_layers = random.randint(1, 4)
        max_neurons = min(32, 4 * number_of_features)
        hidden_layers = sorted(
            [random.randint(number_of_features, max_neurons) for _ in range(number_of_layers)],
            reverse=True
        )
        dropout = random.choice([0.2, 0.3, 0.4, 0.5])
        activation_function = random.choice([
            nn.ReLU,
            nn.Sigmoid,
            nn.Tanh,
            nn.LeakyReLU,
            nn.ELU,
            nn.GELU,
            nn.SELU,
            nn.Softplus
        ])
        return {
            "neural_network_hidden_layers": hidden_layers,
            "dropout": dropout,
            "activation": activation_function
        }
    
    # 4.2 Generate n samples
    random_configs = [generate_random_config(number_of_features) for _ in range(n_samples)]
    
    # 4.3 Apply Random search on Base NN Architecture
    result_random_search = []
    
    for random_param in random_configs:
        fnn = FlexibleNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)
        fnn_model, fnn_train_losses, fnn_val_losses, fnn_best_val, fnn_test_loss, fnn_architecture, fnn_weights = fnn.train_model(
                hidden_layers=random_param['neural_network_hidden_layers'],
                activation_fn=random_param['activation'],
                dropout_rate=random_param['dropout'],
                use_batchnorm=True,
                epochs=epochs_nas,
                early_stopping_patience=tries_before_stopping,
                optimizer_kwargs={"lr": learning_rate}
            )
        fnn_model.to(device)
        predicted_nn = fnn_model(X_test_tensor).detach().cpu().numpy()
        mse_nn = mean_squared_error(y_test, predicted_nn)
        result_random_search.append(mse_nn)
    
    # 4.4 Best result
    best_architecture = result_random_search.index(min(result_random_search))
    
    # 4.5 Best Parameters
    neural_network_hidden_layers = random_configs[best_architecture]['neural_network_hidden_layers']
    dropout_rt = random_configs[best_architecture]['dropout']
    activation_fn = random_configs[best_architecture]['activation']
    
    # =============================================================================
    # 5 - Basic Neural Network
    # =============================================================================
    # Initialize model and train on device
    fnn = FlexibleNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)
    
    fnn_model, fnn_train_losses, fnn_val_losses, fnn_best_val, fnn_test_loss, fnn_architecture, fnn_weights = fnn.train_model(
    hidden_layers=neural_network_hidden_layers,
    activation_fn=activation_fn,
    dropout_rate=dropout_rt,
    use_batchnorm=True,
    epochs=epochs,
    early_stopping_patience=tries_before_stopping,
    optimizer_kwargs={"lr": learning_rate}
    )
    
    # Ensure model is on device
    fnn_model.to(device)

    # =============================================================================
    # 6 - Cobb Neural Networks Random Config Params and Random Search for Cobb NN
    # =============================================================================
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
    random_configs = [generate_random_config(number_of_features) for _ in range(n_samples)]
    
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
    
    # =============================================================================
    # 7 - Cobb-Douglas Neural Network
    # =============================================================================
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

    # =============================================================================
    # 8 - Accuracy
    # =============================================================================
    # X_test_tensor is already on device
    predicted_cobb_nn = cobb_model(X_test_tensor).detach().cpu().numpy()
    predicted_nn = fnn_model(X_test_tensor).detach().cpu().numpy()
    
    # Mean Absolute Error
    mae_nn = mean_absolute_error(y_test, predicted_nn)
    mae_cobb_nn = mean_absolute_error(y_test, predicted_cobb_nn)
    
    # Mean Squared Error
    mse_nn = mean_squared_error(y_test, predicted_nn)
    mse_cobb_nn = mean_squared_error(y_test, predicted_cobb_nn)

    # =============================================================================
    # 9 - Sensitivity Analysis
    # =============================================================================
    # Cobb-Douglas Impact calculation
    cobb_layers = model_alphas(cobb_weights)
    
    cobb_douglas_feature_impact = [
        cobb_douglas_log_impact(cobb_layers, i, normalize=False)
        for i in range(cobb_layers[0].shape[1])
    ]
    
    cobb_douglas_normalized_impact = [
        cobb_douglas_log_impact(cobb_layers, i, normalize=True)
        for i in range(cobb_layers[0].shape[1])
    ]
    
    # Neural Network Impact calculation
    neural_network_feature_impact = []
    
    for i in range(number_of_features):
        x_test_sensitivity_analysis = X_test_tensor.clone()
        x_test_sensitivity_analysis[:, i] *= 1.01
    
        base_predicted_nn = fnn_model(X_test_tensor)
        predicted_nn_sensitivity_analysis = fnn_model(x_test_sensitivity_analysis)
        
        diff = torch.abs(predicted_nn_sensitivity_analysis - base_predicted_nn)
        mae = diff.mean()
        neural_network_feature_impact.append(mae.item())
    
    # SHAP values percentage impact
    shap_impact_percent = compute_shap_impact_percent(fnn_model, X_train, X_test, y_train)
    
    # CAPTUM algorithms
    ig = IntegratedGradients(fnn_model)
    ig_nt = NoiseTunnel(ig)
    dl = DeepLift(fnn_model)
    gs = GradientShap(fnn_model)
    fa = FeatureAblation(fnn_model)

    ig_attr_test = ig.attribute(X_test_tensor, n_steps=50)
    ig_nt_attr_test = ig_nt.attribute(X_test_tensor)
    dl_attr_test = dl.attribute(X_test_tensor)
    gs_attr_test = gs.attribute(X_test_tensor, X_train_tensor)
    fa_attr_test = fa.attribute(X_test_tensor)
    
    # Integrated Gradients
    importances_ig = ig_attr_test.abs().mean(dim=0) 
    importances_ig = importances_ig.detach().cpu().numpy()
    
    # Deeplift
    importances_dl = dl_attr_test.abs().mean(dim=0) 
    importances_dl = importances_dl.detach().cpu().numpy()
    
    # GradientShap
    importances_gs = gs_attr_test.abs().mean(dim=0) 
    importances_gs = importances_gs.detach().cpu().numpy()
    
    # Feature Ablation
    importances_fa = fa_attr_test.abs().mean(dim=0) 
    importances_fa = importances_fa.detach().cpu().numpy()
    
    # Noise Tunnel
    importances_nt = ig_nt_attr_test.abs().mean(dim=0) 
    importances_nt = importances_nt.detach().cpu().numpy()

    # =============================================================================
    # 10 - Results Aggregation
    # =============================================================================
    cobb_douglas_feature_impact = np.array(cobb_douglas_feature_impact, dtype=float)
    neural_network_feature_impact = np.array(neural_network_feature_impact, dtype=float)
    shap_impact_percent = np.array(shap_impact_percent, dtype=float).ravel()
    
    # Coob Douglas Results
    cobb_spearman = compare_importance_spearman(real_feature_impact, cobb_douglas_feature_impact)
    cobb_cossine = compare_importance_cosine(real_feature_impact, cobb_douglas_feature_impact)
    cobb_topk_pair = topk_pairwise_rank_agreement(real_feature_impact, cobb_douglas_feature_impact,number_of_features)
    cobb_topk_rank = topk_rank_correlation(real_feature_impact, cobb_douglas_feature_impact,number_of_features)
    
    # Random Coef Results
    random_spearman = compare_importance_spearman(real_feature_impact, random_feature_impact)
    random_cossine = compare_importance_cosine(real_feature_impact, random_feature_impact)
    random_topk_pair = topk_pairwise_rank_agreement(real_feature_impact, random_feature_impact,number_of_features)
    random_topk_rank = topk_rank_correlation(real_feature_impact, random_feature_impact,number_of_features)
    
    # Neural Network Results
    # Basic Impact
    nn_spearman = compare_importance_spearman(real_feature_impact, neural_network_feature_impact)
    nn_cossine = compare_importance_cosine(real_feature_impact, neural_network_feature_impact)
    nn_topk_pair = topk_pairwise_rank_agreement(real_feature_impact, neural_network_feature_impact,number_of_features)
    nn_topk_rank = topk_rank_correlation(real_feature_impact, neural_network_feature_impact,number_of_features)
    
    # Shap Value
    nn_shap_spearman = compare_importance_spearman(real_feature_impact, shap_impact_percent)
    nn_shap_cossine = compare_importance_cosine(real_feature_impact, shap_impact_percent)
    nn_shap_topk_pair = topk_pairwise_rank_agreement(real_feature_impact, shap_impact_percent,number_of_features)
    nn_shap_topk_rank = topk_rank_correlation(real_feature_impact, shap_impact_percent,number_of_features)
    
    # Integrated Gradient
    nn_shap_spearman_ig = compare_importance_spearman(real_feature_impact, importances_ig)
    nn_shap_cossine_ig = compare_importance_cosine(real_feature_impact, importances_ig)
    nn_shap_topk_pair_ig = topk_pairwise_rank_agreement(real_feature_impact, importances_ig,number_of_features)
    nn_shap_topk_rank_ig = topk_rank_correlation(real_feature_impact, importances_ig,number_of_features)
    
    # Deeplift
    nn_shap_spearman_dl = compare_importance_spearman(real_feature_impact, importances_dl)
    nn_shap_cossine_dl = compare_importance_cosine(real_feature_impact, importances_dl)
    nn_shap_topk_pair_dl = topk_pairwise_rank_agreement(real_feature_impact, importances_dl,number_of_features)
    nn_shap_topk_rank_dl = topk_rank_correlation(real_feature_impact, importances_dl,number_of_features)
    
    # GradientShap
    nn_shap_spearman_gs = compare_importance_spearman(real_feature_impact, importances_gs)
    nn_shap_cossine_gs = compare_importance_cosine(real_feature_impact, importances_gs)
    nn_shap_topk_pair_gs = topk_pairwise_rank_agreement(real_feature_impact, importances_gs,number_of_features)
    nn_shap_topk_rank_gs = topk_rank_correlation(real_feature_impact, importances_gs,number_of_features)
    
    # Feature Ablation
    nn_shap_spearman_fa = compare_importance_spearman(real_feature_impact, importances_fa)
    nn_shap_cossine_fa = compare_importance_cosine(real_feature_impact, importances_fa)
    nn_shap_topk_pair_fa = topk_pairwise_rank_agreement(real_feature_impact, importances_fa,number_of_features)
    nn_shap_topk_rank_fa = topk_rank_correlation(real_feature_impact, importances_fa,number_of_features)
    
    # Noise Tunnel
    nn_shap_spearman_nt = compare_importance_spearman(real_feature_impact, importances_nt)
    nn_shap_cossine_nt = compare_importance_cosine(real_feature_impact, importances_nt)
    nn_shap_topk_pair_nt = topk_pairwise_rank_agreement(real_feature_impact, importances_nt,number_of_features)
    nn_shap_topk_rank_nt = topk_rank_correlation(real_feature_impact, importances_nt,number_of_features)      

    # Store results   
    # Time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    lst_result = [
    # Métricas básicas
    ('mae_nn', mae_nn),
    ('mae_cobb_nn', mae_cobb_nn),
    ('mse_nn', mse_nn),
    ('mse_cobb_nn', mse_cobb_nn),

    # Cobb Douglas Results
    ('cobb_spearman', cobb_spearman),
    ('cobb_cossine', cobb_cossine),
    ('cobb_topk_pair', cobb_topk_pair),
    ('cobb_topk_rank', cobb_topk_rank),
    
    # Random Assigment Results
    ('random_spearman', random_spearman),
    ('random_cossine', random_cossine),
    ('random_topk_pair', random_topk_pair),
    ('random_topk_rank', random_topk_rank),

    # Neural Network Results - Basic Impact
    ('nn_spearman', nn_spearman),
    ('nn_cossine', nn_cossine),
    ('nn_topk_pair', nn_topk_pair),
    ('nn_topk_rank', nn_topk_rank),

    # Neural Network Results - SHAP
    ('nn_shap_spearman', nn_shap_spearman),
    ('nn_shap_cossine', nn_shap_cossine),
    ('nn_shap_topk_pair', nn_shap_topk_pair),
    ('nn_shap_topk_rank', nn_shap_topk_rank),

    # Neural Network Results - Integrated Gradients
    ('nn_shap_spearman_ig', nn_shap_spearman_ig),
    ('nn_shap_cossine_ig', nn_shap_cossine_ig),
    ('nn_shap_topk_pair_ig', nn_shap_topk_pair_ig),
    ('nn_shap_topk_rank_ig', nn_shap_topk_rank_ig),

    # Neural Network Results - DeepLIFT
    ('nn_shap_spearman_dl', nn_shap_spearman_dl),
    ('nn_shap_cossine_dl', nn_shap_cossine_dl),
    ('nn_shap_topk_pair_dl', nn_shap_topk_pair_dl),
    ('nn_shap_topk_rank_dl', nn_shap_topk_rank_dl),

    # Neural Network Results - GradientSHAP
    ('nn_shap_spearman_gs', nn_shap_spearman_gs),
    ('nn_shap_cossine_gs', nn_shap_cossine_gs),
    ('nn_shap_topk_pair_gs', nn_shap_topk_pair_gs),
    ('nn_shap_topk_rank_gs', nn_shap_topk_rank_gs),

    # Neural Network Results - Feature Ablation
    ('nn_shap_spearman_fa', nn_shap_spearman_fa),
    ('nn_shap_cossine_fa', nn_shap_cossine_fa),
    ('nn_shap_topk_pair_fa', nn_shap_topk_pair_fa),
    ('nn_shap_topk_rank_fa', nn_shap_topk_rank_fa),

    # Neural Network Results - Noise Tunnel
    ('nn_shap_spearman_nt', nn_shap_spearman_nt),
    ('nn_shap_cossine_nt', nn_shap_cossine_nt),
    ('nn_shap_topk_pair_nt', nn_shap_topk_pair_nt),
    ('nn_shap_topk_rank_nt', nn_shap_topk_rank_nt),

    # Metadados
    ('elapsed_time', elapsed_time),
    ('number_of_features', number_of_features),
    ('neural_network_hidden_layers', neural_network_hidden_layers),
    ('activition_function', activition_function),
    ('dropout', dropout)
    ]

    print(f"[{i+1}/{sim_epochs}] "
          f"MAE_NN={mae_nn:.4f} | MSE_NN={mse_nn:.4f} | "
          f"Features={number_of_features} | "
          f"Time={elapsed_time:.2f}s")

    result_list.append(lst_result)
    
# =============================================================================
# 11 - Final Results WITH CONFIDENCE INTERVALS
# =============================================================================
df, df_cosine_ci, df_spearman_ci, df_mae_ci, df_mse_ci, df_topk_pair_ci, df_topk_rank_ci = generate_average_metrics_with_ci(result_list)

# Save results with CI
wkdir = "/home/jeronymo/Downloads/neural_cge_network/src/data_simulation/Results/Experiment_" + str(experiment_number)
os.chdir(wkdir)

df.to_csv('simulation_result.csv', index=False)
df_cosine_ci.to_csv('df_cosine_ci_result.csv', index=False)
df_spearman_ci.to_csv('df_spearman_ci_result.csv', index=False)
df_topk_pair_ci.to_csv('df_topk_pair_ci_result.csv', index=False)
df_topk_rank_ci.to_csv('df_topk_rank_ci_result.csv', index=False)
df_mae_ci.to_csv('df_mae_ci_result.csv', index=False)
df_mse_ci.to_csv('df_mse_ci_result.csv', index=False)

# =============================================================================
# =============================================================================
