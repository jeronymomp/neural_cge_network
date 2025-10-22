#--------
# Working Directory
import os
wkdir = "/home/jeronymo/Documentos/neural_cge_network"
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
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
random_state = 0
n_samples = 30
g = 0.02  
δ = 0.03  

#--------
# 2.1 - Experiment Parameters
# Starting and ending year
start_year = 1971
end_year = 2019
n_years = end_year - start_year

#--------
# Neural Network Parameters
epochs_nas = 5000
epochs = 20000
# neural_network_hidden_layers = [4,3]
learning_rate = 0.00001
learning_rate_cobb = 0.001
tries_before_stopping = 5

#--------
# Cobb-Douglas Neural Network Parameters
sum_equal_one = 1

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
# 3 - Import PWT Data
# =============================================================================
# 3.1 - OS
wkdir = "/home/jeronymo/Documentos/neural_cge_network/src/empirical_application"
os.chdir(wkdir)

# 3.2 - Country COdes
variables = ["countrycode","year","rgdpna","pop","rnna"]
start_year = 1971  # ajustar conforme necessário

pwt_data = pd.read_excel('pwt_data.xlsx')
df = pwt_data[variables]

# 3.3 - Filter
df = df[df['year'] >= start_year]

# 3.4 - Y per capita
df["Y_per_L"] = df["rgdpna"] / df["pop"]

# 3.5 -  Pop Growth
df_period = pwt_data[(pwt_data["year"] >= 1960) & (pwt_data["year"] <= 1985)].copy()
pop_growth = (
    df_period.groupby("countrycode")["pop"]
    .agg(["first", "last"])
    .reset_index()
)
pop_growth["n"] = (np.log(pop_growth["last"]) - np.log(pop_growth["first"])) / (1985 - 1960)

# 3.6 - Investment
inv_mean = (
    df_period.groupby("countrycode")["rnna"]
    .mean()
    .reset_index()
    .rename(columns={"rnna": "s"})
)

# 3.7 - data Prep
reg_df = df[["countrycode","Y_per_L"]].merge(pop_growth[["countrycode","n"]], on="countrycode")
reg_df = reg_df.merge(inv_mean, on="countrycode")

reg_df = reg_df[(reg_df["Y_per_L"] > 0) & (reg_df["s"] > 0) & (reg_df["n"] > 0)]
reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna()

scaler_n = MinMaxScaler()
scaler_s = MinMaxScaler()
scaler_y = MinMaxScaler()

reg_df['n'] = reg_df['n'] + g + δ

reg_df["ln_Y_per_L"] = np.log(reg_df[["Y_per_L"]])
reg_df["ln_s"] = np.log(reg_df[["s"]])
reg_df["ln_n"] = np.log(reg_df[["n"]]) 

reg_df["scaled_Y_per_L"] = scaler_y.fit_transform(reg_df[["ln_Y_per_L"]])
reg_df["scaled_s"] = scaler_s.fit_transform(reg_df[["ln_s"]])
reg_df["scaled_n"] = scaler_n.fit_transform(reg_df[["ln_n"]]) 

reg_df = reg_df.groupby('countrycode', as_index=False).mean()

X = reg_df[["s","n"]]
y = reg_df["ln_Y_per_L"]

ln_X = reg_df[["scaled_s","scaled_n"]]
ln_y = reg_df["scaled_Y_per_L"]

# 3.8 Prepare data. Only IN-Sample - For Cobb DOuglas
y_full = y.values.reshape(-1, 1).astype(np.float32)
X_full = X.values.astype(np.float32)

X_tensor = torch.from_numpy(X_full).to(device)
y_tensor = torch.from_numpy(y_full).to(device)

X_train = X_full.copy()
X_valid = X_full.copy()
X_test  = X_full.copy()

y_train = y_full.copy()
y_valid = y_full.copy()
y_test  = y_full.copy()

X_train_tensor = X_tensor
X_valid_tensor = X_tensor
X_test_tensor  = X_tensor

y_train_tensor = y_tensor.reshape(-1, 1)
y_valid_tensor = y_tensor.reshape(-1, 1)
y_test_tensor  = y_tensor.reshape(-1, 1)

# 3.9 Prepare data. Only IN-Sample - For FNN
ln_y_full = ln_y.values.reshape(-1, 1).astype(np.float32)
ln_X_full = ln_X.values.astype(np.float32)

ln_X_tensor = torch.from_numpy(ln_X_full).to(device)
ln_y_tensor = torch.from_numpy(ln_y_full).to(device)

ln_X_train = ln_X_full.copy()
ln_X_valid = ln_X_full.copy()
ln_X_test  = ln_X_full.copy()

ln_y_train = ln_y_full.copy()
ln_y_valid = ln_y_full.copy()
ln_y_test  = ln_y_full.copy()

ln_X_train_tensor = ln_X_tensor
ln_X_valid_tensor = ln_X_tensor
ln_X_test_tensor  = ln_X_tensor

ln_y_train_tensor = ln_y_tensor.reshape(-1, 1)
ln_y_valid_tensor = ln_y_tensor.reshape(-1, 1)
ln_y_test_tensor  = ln_y_tensor.reshape(-1, 1)

# Number of features
number_of_features = len(X.columns)

# =============================================================================
# 4 - Linear Regression
# =============================================================================
model = LinearRegression()
model.fit(ln_X, ln_y.values.reshape(-1,1))   

y_pred = model.predict(ln_X)
mse_regression = mean_squared_error(ln_y.values.reshape(-1,1), y_pred)
mae_regression = mean_absolute_error(ln_y.values.reshape(-1,1), y_pred)
r2_regression = r2_score(ln_y.values.reshape(-1,1), y_pred)

# =============================================================================
# 5 - Neural Networks Random Config Params and Random Search for Base NN
# =============================================================================
# 5.1 Function to perform random search
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

# 5.2 Generate n samples
random_configs = [generate_random_config(number_of_features) for _ in range(n_samples)]

# 5.3 Apply Random search on Base NN Architecture
result_random_search = []

for random_param in random_configs:
    fnn = FlexibleNeuralNetwork(ln_X_train, ln_X_valid, ln_X_test, ln_y_train, ln_y_valid, ln_y_test, device = device)
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
    predicted_nn = fnn_model(ln_X_test_tensor).detach().cpu().numpy()
    mse_nn = mean_squared_error(ln_y_test, predicted_nn)
    result_random_search.append(mse_nn)

# 5.4 Best result
best_architecture = result_random_search.index(min(result_random_search))

# 5.5 Best Parameters
neural_network_hidden_layers = random_configs[best_architecture]['neural_network_hidden_layers']
dropout_rt = random_configs[best_architecture]['dropout']
activation_fn = random_configs[best_architecture]['activation']

# =============================================================================
# 6 - Basic Neural Network
# =============================================================================
# Initialize model and train on device
fnn = FlexibleNeuralNetwork(ln_X_train, ln_X_valid, ln_X_test, ln_y_train, ln_y_valid, ln_y_test, device = device)

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
# 7 - Cobb Neural Networks Random Config Params and Random Search for Cobb NN
# =============================================================================
# 7.1 Function to perform random search
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

# 7.2 Generate n samples
random_configs = [generate_random_config(number_of_features) for _ in range(n_samples)]

# 7.3 Apply Random search on Base NN Architecture
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
    optimizer_kwargs={"lr": learning_rate_cobb}
    )

    cobb_model.to(device)
    predicted_cobb = cobb_model(X_test_tensor).detach().cpu().numpy()
    mse_cobb = mean_squared_error(y_test, predicted_cobb)
    result_random_search_cobb.append(mse_cobb)

# 7.4 Best result
best_cobb_architecture = result_random_search_cobb.index(min(result_random_search_cobb))

# 7.5 Best Parameters
cobb_hidden_layers = random_configs[best_cobb_architecture]['neural_network_hidden_layers']
cobb_dropout_rt = random_configs[best_cobb_architecture]['dropout']

# =============================================================================
# 8 - Cobb-Douglas Neural Network
# =============================================================================
cobb_fnn = LogCobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)

cobb_model, cobb_train_losses, cobb_val_losses, cobb_best_val_loss, cobb_test_loss, cobb_architecture, cobb_weights = cobb_fnn.train_model(
hidden_layers=cobb_hidden_layers,
dropout_rate=cobb_dropout_rt,
use_batchnorm=False,
early_stopping_patience=tries_before_stopping,
epochs=epochs,
normalized=sum_equal_one,
optimizer_kwargs={"lr": learning_rate_cobb}
)

cobb_model.to(device)

# =============================================================================
# 9 - Accuracy
# =============================================================================
# X_test_tensor is already on device
predicted_cobb_nn = cobb_model(X_test_tensor).detach().cpu().numpy()
predicted_nn = fnn_model(ln_X_test_tensor).detach().cpu().numpy()

# Mean Absolute Error
mae_nn = mean_absolute_error(ln_y_test, predicted_nn)
mae_cobb_nn = mean_absolute_error(y_test, predicted_cobb_nn)

# Mean Squared Error
mse_nn = mean_squared_error(ln_y_test, predicted_nn)
mse_cobb_nn = mean_squared_error(y_test, predicted_cobb_nn)

# R Squared
r2_cobb = r2_score(y_test, predicted_cobb_nn)
r2_nn = r2_score(ln_y_test, predicted_nn)

# =============================================================================
# 10 - Sensitivity Analysis
# =============================================================================
# Real Feature Impact
real_feature_impact = compute_linear_regression_feature_impact_percent(ln_X_test, model)

# Random Assigment
min_val = ln_X_train.min()
max_val = ln_X_train.max()
random_feature_impact = np.random.uniform(low=min_val, high=max_val, size=real_feature_impact.shape)

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
    x_test_sensitivity_analysis = ln_X_test_tensor.clone()
    x_test_sensitivity_analysis[:, i] *= 1.01
    
    base_predicted_nn = fnn_model(ln_X_test_tensor)
    predicted_nn_sensitivity_analysis = fnn_model(x_test_sensitivity_analysis)
        
    diff = (predicted_nn_sensitivity_analysis - base_predicted_nn) / base_predicted_nn
    
    mae = diff.mean() / 0.01

    neural_network_feature_impact.append(mae.item())
    
# SHAP values percentage impact
shap_impact_percent = compute_shap_impact_percent(fnn_model, ln_X_train, ln_X_test, ln_y_train)
    
# CAPTUM algorithms
ig = IntegratedGradients(fnn_model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(fnn_model)
gs = GradientShap(fnn_model)
fa = FeatureAblation(fnn_model)

ig_attr_test = ig.attribute(ln_X_test_tensor, n_steps=50)
ig_nt_attr_test = ig_nt.attribute(ln_X_test_tensor)
dl_attr_test = dl.attribute(ln_X_test_tensor)
gs_attr_test = gs.attribute(ln_X_test_tensor, ln_X_train_tensor)
fa_attr_test = fa.attribute(ln_X_test_tensor)
    
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
# 11 - Results Aggregation
# =============================================================================
cobb_douglas_feature_impact = np.array(cobb_douglas_feature_impact, dtype=float)
neural_network_feature_impact = np.array(neural_network_feature_impact, dtype=float)
shap_impact_percent = np.array(shap_impact_percent, dtype=float).ravel()

# Taken from literature
real_feature_impact = np.array([1.42, -1.97])

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
]

result_list.append(lst_result)

# =============================================================================
# 11 - Final Results WITH CONFIDENCE INTERVALS
# =============================================================================
df, df_cosine_ci, df_spearman_ci, df_mae_ci, df_mse_ci, df_topk_pair_ci, df_topk_rank_ci = generate_average_metrics_with_ci(result_list)

# Save results
wkdir = "/home/jeronymo/Documentos/neural_cge_network/src/empirical_application/Solow/Results/Experiment 1"
os.chdir(wkdir)

df.to_csv('simulation_result.csv', index=False)
df_cosine_ci.to_csv('df_cosine_mean_result.csv', index=False)
df_spearman_ci.to_csv('df_spearman_mean_result.csv', index=False)
df_mae_ci.to_csv('df_mae_mean_result.csv', index=False)
df_mse_ci.to_csv('df_mse_mean_result.csv', index=False)
df_topk_pair_ci.to_csv('df_topk_pair_mean_result.csv', index=False)
df_topk_rank_ci.to_csv('df_topk_rank_mean_result.csv', index=False)

# Coef DataFrame
real_feature_impact = np.array(real_feature_impact).flatten()
cobb_douglas_feature_impact = np.array(cobb_douglas_feature_impact).flatten()
neural_network_feature_impact = np.array(neural_network_feature_impact).flatten()
shap_impact_percent = np.array(shap_impact_percent).flatten()
importances_ig = np.array(importances_ig).flatten()
importances_dl = np.array(importances_dl).flatten()
importances_gs = np.array(importances_gs).flatten()
importances_fa = np.array(importances_fa).flatten()
importances_nt = np.array(importances_nt).flatten()

df_coef = pd.DataFrame({
    "real_feature_impact": real_feature_impact,
    "cobb_douglas_feature_impact": cobb_douglas_feature_impact,
    "neural_network_feature_impact": neural_network_feature_impact,
    "shap_impact_percent": shap_impact_percent,
    "importances_ig": importances_ig,
    "importances_dl": importances_dl,
    "importances_gs": importances_gs,
    "importances_fa": importances_fa,
    "importances_nt": importances_nt
})

df_coef.to_csv("coefficients.csv", index=False)

# =============================================================================
# =============================================================================
