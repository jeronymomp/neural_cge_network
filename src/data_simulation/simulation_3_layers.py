#--------
# Working Directory
import os
wkdir = "/home/jeronymo/Documentos/neural_cge_network"
os.chdir(wkdir)

# =============================================================================
# 1 - Packages
# =============================================================================
# External Packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Built-in
from src.functions.linear_data_generator import LinearDataGenerator
from src.functions.feature_impact import compute_linear_regression_feature_impact_percent,cosine_similarity
from src.functions.feature_impact import cobb_douglas_impact,model_alphas,compute_shap_impact_percent
from src.functions.neural_network_function import FlexibleNeuralNetwork
from src.functions.cobb_douglas_act_function import CobbDouglasActivationFunction
from src.functions.cobb_doulgas_neural_network_function import CobbDouglasNeuralNetwork

# =============================================================================
# 2 - Parameters
# =============================================================================
#--------
# 2.1 - Experiment Parameters
sample_size = 1000
number_of_features = 5
random_state = 0
test_sample_size = 0.2
valid_sample_size = 0.2

#--------
# Neural Network Parameters
epochs = 1000
neural_network_hidden_layers = [4,3]
dropout = 0.2
learning_rate = 0.001
tries_before_stopping = 10

#--------
# Cobb-Douglas Neural Network Parameters
sum_equal_one = 0

# =============================================================================
# 3 - Data Generation and Preparation
# =============================================================================
# Generate Simulated Data
linear_data_generator = LinearDataGenerator(sample_size,number_of_features,random_state)
X, y, coef = linear_data_generator.generate_linear_regression_data()

X = np.abs(X)
y = np.abs(y)

# Normalization -  Optional
# scaler = StandardScaler()
#X = scaler.fit_transform(X)

# Percentual Impact Impact
linear_model = LinearRegression().fit(X, y)
real_feature_impact = compute_linear_regression_feature_impact_percent(X, linear_model)

# Train Test Split
# Test x Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sample_size, random_state=random_state)
# Valid x Train
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_sample_size, random_state=random_state)

# =============================================================================
# 4 - Basic Neural Network
# =============================================================================
fnn = FlexibleNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test)

fnn_model, fnn_train_losses, fnn_val_losses, fnn_best_val, fnn_test_loss, fnn_architecture, fnn_weights = fnn.train_model(
    hidden_layers=neural_network_hidden_layers,
    activation_fn=nn.LeakyReLU,
    dropout_rate=dropout,
    use_batchnorm=True,
    epochs=epochs,
    early_stopping_patience=tries_before_stopping,
    optimizer_kwargs={"lr": learning_rate}
)

# =============================================================================
# 5 - Cobb-Douglas Neural Network
# =============================================================================
cobb_fnn = CobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test)

cobb_model, cobb_train_losses, cobb_val_losses, cobb_best_val_loss, cobb_test_loss, cobb_architecture, cobb_weights = cobb_fnn.train_model(
    hidden_layers=neural_network_hidden_layers,
    dropout_rate=dropout,
    use_batchnorm=False,
    early_stopping_patience=tries_before_stopping,
    epochs=epochs,
    normalized=sum_equal_one,
    optimizer_kwargs={"lr": learning_rate}
)

# =============================================================================
# 6 - Accuracy
# =============================================================================
X_test_tensor = torch.from_numpy(X_test).float()
predicted_cobb_nn = cobb_model(X_test_tensor).detach().cpu().numpy()
predicted_nn = fnn_model(X_test_tensor).detach().cpu().numpy()

# Mean Absolute Error
mae_nn = mean_absolute_error(y_test, predicted_nn)
mae_cobb_nn = mean_absolute_error(y_test, predicted_cobb_nn)

# Mean Squared Error
mse_nn = mean_squared_error(y_test, predicted_nn)
mse_cobb_nn = mean_squared_error(y_test, predicted_cobb_nn)

# =============================================================================
# 7 - Sensitivity Analysis
# =============================================================================
#----------------
# 7.1 - Cobb-Douglas Impact
cobb_layers = model_alphas(cobb_weights)

# Perc Impact (%)
cobb_douglas_feature_impact = [
    cobb_douglas_impact(cobb_layers, i, normalize=False)
    for i in range(cobb_layers[0].shape[1])
]

# Normalized Impact
cobb_douglas_normalized_impact = [
    cobb_douglas_impact(cobb_layers, i, normalize=True)
    for i in range(cobb_layers[0].shape[1])
]

#----------------------
# 7.2 - Neural Network Impact
neural_network_feature_impact = []

for i in range(number_of_features):
    x_test_sensitivity_analysis = X_test_tensor.clone()
    x_test_sensitivity_analysis[:, i] *= 1.01

    # Predict
    base_predicted_nn = fnn_model(X_test_tensor)
    predicted_nn_sensitivity_analysis = fnn_model(x_test_sensitivity_analysis)
    
    diff = torch.abs(predicted_nn_sensitivity_analysis - base_predicted_nn)
    mae = diff.mean()
    neural_network_feature_impact.append(mae.item())

#----------------------
# 7.3 - % Shap Values
shap_impact_percent = compute_shap_impact_percent(fnn_model, X_train, X_test, y_train)

# =============================================================================
# 7 - Results
# =============================================================================
# 7.1 - Impact
print(f"Impacto real percentual médio da feature é de {real_feature_impact}")
print(f"Impacto cobb-douglas percentual médio da feature é de {cobb_douglas_feature_impact}")
print(f"Impacto neural-network percentual médio da feature é de {neural_network_feature_impact}")

# 7.2 - Accuracy
print(f"O MSE da rede neural básica é de {mse_nn} e o MAE de {mae_nn}")
print(f"O MSE da rede neural cobb-douglas é de {mse_cobb_nn} e o MAE de {mae_cobb_nn}")

similarity_cobb_douglas = cosine_similarity(real_feature_impact, cobb_douglas_feature_impact)
similarity_neural_network = cosine_similarity(real_feature_impact, neural_network_feature_impact)
similarity_neural_network_shap = cosine_similarity(real_feature_impact, shap_impact_percent)
similarity_normalized_cobb_douglas = cosine_similarity(real_feature_impact, cobb_douglas_normalized_impact)

# =============================================================================
# =============================================================================