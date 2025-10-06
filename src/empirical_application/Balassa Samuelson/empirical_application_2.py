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
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

# Built-in
from src.functions.linear_data_generator import LinearDataGenerator
from src.functions.feature_impact import compute_linear_regression_feature_impact_percent
from src.functions.feature_impact import cobb_douglas_impact,cobb_douglas_log_impact,model_alphas,compute_shap_impact_percent
from src.functions.feature_impact import compare_importance_cosine,normalize_importance,compare_importance_spearman
from src.functions.neural_network_function import FlexibleNeuralNetwork
from src.functions.cobb_douglas_act_function import CobbDouglasActivationFunction
from src.functions.log_cobb_douglas_act_function import CobbDouglasLogActivationFunction
from src.functions.cobb_doulgas_neural_network_function import CobbDouglasNeuralNetwork
from src.functions.log_cobb_doulgas_neural_network_function import LogCobbDouglasNeuralNetwork
from src.functions.result_analysis import generate_average_metrics_separated

# =============================================================================
# 2 - Parameters
# =============================================================================
random_state = 0
test_sample_size = 0.1
valid_sample_size = 0.1
n_samples = 200

#--------
# 2.1 - Experiment Parameters
# Starting and ending year
start_year = 1971
end_year = 2019
n_years = end_year - start_year

# The variable are:
    # rgdpo	- Output-side real GDP at chained PPPs (in mil. 2017US$)
    # pop - Population (in millions)
    # pl_c - Price level of household consumption,  price level of USA GDPo in 2017=1
# Existing Variables
variables = ['countrycode','year','rgdpo','pl_c','pop']
# New Variables
gdp_per_capita_variable_name = 'gdp_percapita'
gdp_percapita_2_variable_name = 'gdp_percapita_2'
price_level_variable_name = 'pl_c'

# Set of variables from regression
y_experiment_2 = [price_level_variable_name]
x_experiment_2 = [gdp_per_capita_variable_name,gdp_percapita_2_variable_name]
number_of_features = len(x_experiment_2)

#--------
# Neural Network Parameters
epochs_nas = 2000
epochs = 20000
# neural_network_hidden_layers = [4,3]
learning_rate = 0.0001
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
wkdir = "/home/jeronymo/Documentos/neural_cge_network/src/empirical_application"
os.chdir(wkdir)

# 3.1 Import data
pwt_data = pd.read_excel('pwt_data.xlsx')
base_df = pwt_data[variables]

# 3.2 Set starting year as 1971
base_df = base_df[base_df['year']>=start_year]

# 3.3 Create Variables
# GDP Per capita
base_df['gdp_percapita'] = base_df['rgdpo'] / base_df['pop']

# 3.4 Country-Code Mean
experiment_2_df = base_df.groupby('countrycode')[['pl_c', 'gdp_percapita']].mean().reset_index()

scaler = MinMaxScaler(feature_range=(0, 1))
experiment_2_df['gdp_percapita'] = scaler.fit_transform(experiment_2_df[['gdp_percapita']])
experiment_2_df['gdp_percapita_2'] = pow(experiment_2_df['gdp_percapita'],2)
experiment_2_df = experiment_2_df.dropna()

# 3.6 Train Test Split (with numpy)
y = pd.DataFrame(experiment_2_df[price_level_variable_name])
x = pd.DataFrame(experiment_2_df[[gdp_per_capita_variable_name,gdp_percapita_2_variable_name]])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_sample_size, random_state=random_state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_sample_size, random_state=random_state)

# 3.7 Adjust to numpy array for FNN
X_train_tensor = torch.from_numpy(X_train.values).float().to(device)
X_valid_tensor = torch.from_numpy(X_valid.values).float().to(device)
X_test_tensor  = torch.from_numpy(X_test.values).float().to(device)

y_train_tensor = torch.from_numpy(y_train.values).float().to(device)
y_valid_tensor = torch.from_numpy(y_valid.values).float().to(device)
y_test_tensor  = torch.from_numpy(y_test.values).float().to(device)


# =============================================================================
# 4 - Linear Regression
# =============================================================================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse_regression = mean_squared_error(y_test, y_pred)
mae_regression = mean_absolute_error(y_test, y_pred)
r2_regression = r2_score(y_test, y_pred)

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
    fnn = FlexibleNeuralNetwork(X_train.values, X_valid.values, X_test.values, y_train.values, y_valid.values, y_test.values, device = device)
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
fnn = FlexibleNeuralNetwork(X_train.values, X_valid.values, X_test.values, y_train.values, y_valid.values, y_test.values, device = device)

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
    cobb_fnn = LogCobbDouglasNeuralNetwork(X_train.values, X_valid.values, X_test.values, y_train.values, y_valid.values, y_test.values, device = device)

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

# 7.4 Best result
best_cobb_architecture = result_random_search_cobb.index(min(result_random_search_cobb))

# 7.5 Best Parameters
cobb_hidden_layers = random_configs[best_cobb_architecture]['neural_network_hidden_layers']
cobb_dropout_rt = random_configs[best_cobb_architecture]['dropout']

# =============================================================================
# 8 - Cobb-Douglas Neural Network
# =============================================================================
cobb_fnn = LogCobbDouglasNeuralNetwork(X_train.values, X_valid.values, X_test.values, y_train.values, y_valid.values, y_test.values, device = device)

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
# 9 - Accuracy
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

# R Squared
r2_cobb = r2_score(y_test, predicted_cobb_nn)
r2_nn = r2_score(y_test, predicted_nn)

# =============================================================================
# 10 - Sensitivity Analysis
# =============================================================================
# Real Feature Impact
real_feature_impact = compute_linear_regression_feature_impact_percent(X_test, model)

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
        
    diff = (predicted_nn_sensitivity_analysis - base_predicted_nn) / base_predicted_nn
    
    mae = diff.mean() / 0.01

    neural_network_feature_impact.append(mae.item())
    
# SHAP values percentage impact
shap_impact_percent = compute_shap_impact_percent(fnn_model, X_train.values, X_test.values, y_train.values)
    
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
# 11 - Results Aggregation
# =============================================================================
a_norm = normalize_importance(real_feature_impact)
b_norm = normalize_importance(cobb_douglas_feature_impact)
c_norm = normalize_importance(neural_network_feature_impact)

cobb_douglas_feature_impact = np.array(cobb_douglas_feature_impact).reshape(-1, 1)
neural_network_feature_impact = np.array(neural_network_feature_impact).reshape(-1, 1)

# Coob Douglas Results
cobb_spearman = compare_importance_spearman(real_feature_impact, cobb_douglas_feature_impact,True)
cobb_cossine = compare_importance_cosine(real_feature_impact, cobb_douglas_feature_impact,True)

# Neural Network Results
# Basic Impact
nn_spearman = compare_importance_spearman(real_feature_impact, neural_network_feature_impact,True)
nn_cossine = compare_importance_cosine(real_feature_impact, neural_network_feature_impact,True)
# Shap Value
nn_shap_spearman = compare_importance_spearman(real_feature_impact, shap_impact_percent,True)
nn_shap_cossine = compare_importance_cosine(real_feature_impact, shap_impact_percent,True)
# Integrated Gradient
nn_shap_spearman_ig = compare_importance_spearman(real_feature_impact, importances_ig,True)
nn_shap_cossine_ig = compare_importance_cosine(real_feature_impact, importances_ig,True)  
# Deeplift
nn_shap_spearman_dl = compare_importance_spearman(real_feature_impact, importances_dl,True)
nn_shap_cossine_dl = compare_importance_cosine(real_feature_impact, importances_dl,True)  
# GradientShap
nn_shap_spearman_gs = compare_importance_spearman(real_feature_impact, importances_gs,True)
nn_shap_cossine_gs = compare_importance_cosine(real_feature_impact, importances_gs,True)  
# Feature Ablation
nn_shap_spearman_fa = compare_importance_spearman(real_feature_impact, importances_fa,True)
nn_shap_cossine_fa = compare_importance_cosine(real_feature_impact, importances_fa,True)  
# Noise Tunnel
nn_shap_spearman_nt = compare_importance_spearman(real_feature_impact, importances_nt,True)
nn_shap_cossine_nt = compare_importance_cosine(real_feature_impact, importances_nt,True)          
   

# Save as List
result_list = []

r2_cobb = r2_score(y_test, predicted_cobb_nn)
r2_nn = r2_score(y_test, predicted_nn)

lst_result = [
('mae_nn', mae_nn),
('mae_regression', mae_regression),
('mae_cobb_nn', mae_cobb_nn),
('r2_cobb', r2_cobb),
('r2_nn', r2_nn),
('r2_regression', r2_regression),
('mse_nn', mse_nn),
('mse_cobb_nn', mse_cobb_nn),
('mse_regression', mse_regression),
('cobb_spearman', cobb_spearman),
('cobb_cossine', cobb_cossine),
('nn_spearman', nn_spearman),
('nn_cossine', nn_cossine),
('nn_spearman_ig', nn_shap_spearman_ig),
('nn_cossine_ig', nn_shap_cossine_ig),
('nn_shap_spearman', nn_shap_spearman),
('nn_shap_cossine', nn_shap_cossine),
('nn_shap_spearman_dl', nn_shap_spearman_dl),
('nn_shap_cossine_dl', nn_shap_cossine_dl),
('nn_shap_spearman_gs', nn_shap_spearman_gs),
('nn_shap_cossine_gs', nn_shap_cossine_gs),
('nn_shap_spearman_fa', nn_shap_spearman_fa),
('nn_shap_cossine_fa', nn_shap_cossine_fa),
('nn_shap_spearman_nt', nn_shap_spearman_nt),
('nn_shap_cossine_nt', nn_shap_cossine_nt),
('number_of_features', number_of_features),
('neural_network_hidden_layers', neural_network_hidden_layers),
('activition_function', activation_fn),
('dropout', dropout_rt)
]

result_list.append(lst_result)

# R² results
r2_results = [item for item in lst_result if item[0].startswith('r2_')]
df_r2 = pd.DataFrame([dict(r2_results)])

# =============================================================================
# 12 - Final Results
# =============================================================================
df, df_cosine_mean, df_spearman_mean, df_mae_mean, df_mse_mean = generate_average_metrics_separated(result_list)

# Save results
wkdir = "/home/jeronymo/Documentos/neural_cge_network/src/empirical_application/Balassa Samuelson/Results/Experiment 2"
os.chdir(wkdir)

df.to_csv('simulation_result.csv', index=False)
df_cosine_mean.to_csv('df_cosine_mean_result.csv', index=False)
df_spearman_mean.to_csv('df_spearman_mean_result.csv', index=False)
df_mae_mean.to_csv('df_mae_mean_result.csv', index=False)
df_mse_mean.to_csv('df_mse_mean_result.csv', index=False)
df_r2.to_csv('df_r2_mean_result.csv', index=False)

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
