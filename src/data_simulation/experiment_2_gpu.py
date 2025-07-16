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
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Built-in
from src.functions.linear_data_generator import LinearDataGenerator
from src.functions.feature_impact import compute_linear_regression_feature_impact_percent
from src.functions.feature_impact import cobb_douglas_impact,model_alphas,compute_shap_impact_percent
from src.functions.feature_impact import compare_importance_cosine,normalize_importance,compare_importance_spearman
from src.functions.neural_network_function import FlexibleNeuralNetwork
from src.functions.cobb_douglas_act_function import CobbDouglasActivationFunction
from src.functions.cobb_doulgas_neural_network_function import CobbDouglasNeuralNetwork
from src.functions.result_analysis import generate_average_metrics_separated

# =============================================================================
# 2 - Parameters
# =============================================================================
result_list = []

#--------
# 2.1 - Experiment Parameters
sample_size = 10000
number_of_features = 5
random_state = 1
test_sample_size = 0.2
valid_sample_size = 0.2

#--------
# Neural Network Parameters
epochs = 1000
neural_network_hidden_layers = [4,3]
learning_rate = 0.00001
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
# Starting Loop
# =============================================================================
for i in range(0,1000):
    # Start Time
    start_time = time.time()
    # Generate random sample
    random_state = random.randint(0,10000000)
    # Generate Random Number of Features
    number_of_features_total = random.randint(2,5)
    number_of_non_features= random.randrange(1, number_of_features_total)
    number_of_features = number_of_features_total - number_of_non_features
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
    # Generate Simulated Data
    linear_data_generator = LinearDataGenerator(sample_size,number_of_features,random_state)
    X, y, coef = linear_data_generator.generate_linear_regression_data()
    
    X = np.abs(X)
    y = np.abs(y)
    
    # Percentual Impact Impact
    linear_model = LinearRegression().fit(X, y)
    real_feature_impact = compute_linear_regression_feature_impact_percent(X, linear_model)
    
    # New feature(s) without significance
    linear_data_generator_2 = LinearDataGenerator(sample_size,number_of_non_features,(random_state+1))
    X_2, y_2, coef_2 = linear_data_generator_2.generate_linear_regression_data()
    X_2 = np.abs(X_2)
    
    # Append to existing array
    X = np.hstack([X, X_2])
    real_feature_impact = np.append(real_feature_impact, np.zeros(number_of_non_features))
    
    # Train Test Split (with numpy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sample_size, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_sample_size, random_state=random_state)

    # Convert splits to tensors on device (GPU/CPU)
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_valid_tensor = torch.from_numpy(X_valid).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    # =============================================================================
    # 4 - Basic Neural Network
    # =============================================================================
    # Initialize model and train on device
    fnn = FlexibleNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)
    
    fnn_model, fnn_train_losses, fnn_val_losses, fnn_best_val, fnn_test_loss, fnn_architecture, fnn_weights = fnn.train_model(
        hidden_layers=neural_network_hidden_layers,
        activation_fn=nn.LeakyReLU,
        dropout_rate=dropout,
        use_batchnorm=True,
        epochs=epochs,
        early_stopping_patience=tries_before_stopping,
        optimizer_kwargs={"lr": learning_rate}
    )
    
    # Ensure model is on device
    fnn_model.to(device)

    # =============================================================================
    # 5 - Cobb-Douglas Neural Network
    # =============================================================================
    cobb_fnn = CobbDouglasNeuralNetwork(X_train, X_valid, X_test, y_train, y_valid, y_test, device = device)
    
    cobb_model, cobb_train_losses, cobb_val_losses, cobb_best_val_loss, cobb_test_loss, cobb_architecture, cobb_weights = cobb_fnn.train_model(
        hidden_layers=neural_network_hidden_layers,
        dropout_rate=dropout,
        use_batchnorm=False,
        early_stopping_patience=tries_before_stopping,
        epochs=epochs,
        normalized=sum_equal_one,
        optimizer_kwargs={"lr": learning_rate}
    )
    
    cobb_model.to(device)

    # =============================================================================
    # 6 - Accuracy
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
    # 7 - Sensitivity Analysis
    # =============================================================================
    # Cobb-Douglas Impact calculation
    cobb_layers = model_alphas(cobb_weights)
    
    cobb_douglas_feature_impact = [
        cobb_douglas_impact(cobb_layers, i, normalize=False)
        for i in range(cobb_layers[0].shape[1])
    ]
    
    cobb_douglas_normalized_impact = [
        cobb_douglas_impact(cobb_layers, i, normalize=True)
        for i in range(cobb_layers[0].shape[1])
    ]
    
    # Neural Network Impact calculation
    neural_network_feature_impact = []
    
    for i in range(number_of_features_total):
        x_test_sensitivity_analysis = X_test_tensor.clone()
        x_test_sensitivity_analysis[:, i] *= 1.01
    
        base_predicted_nn = fnn_model(X_test_tensor)
        predicted_nn_sensitivity_analysis = fnn_model(x_test_sensitivity_analysis)
        
        diff = torch.abs(predicted_nn_sensitivity_analysis - base_predicted_nn)
        mae = diff.mean()
        neural_network_feature_impact.append(mae.item())
    
    # SHAP values percentage impact
    shap_impact_percent = compute_shap_impact_percent(fnn_model, X_train, X_test, y_train)

    # =============================================================================
    # 8 - Results Aggregation
    # =============================================================================
    a_norm = normalize_importance(real_feature_impact)
    b_norm = normalize_importance(cobb_douglas_feature_impact)
    c_norm = normalize_importance(neural_network_feature_impact)
    
    cobb_spearman = compare_importance_spearman(real_feature_impact, cobb_douglas_feature_impact)
    cobb_cossine = compare_importance_cosine(real_feature_impact, cobb_douglas_feature_impact)
    
    nn_spearman = compare_importance_spearman(real_feature_impact, neural_network_feature_impact)
    nn_cossine = compare_importance_cosine(real_feature_impact, neural_network_feature_impact)
    
    nn_shap_spearman = compare_importance_spearman(real_feature_impact, shap_impact_percent)
    nn_shap_cossine = compare_importance_cosine(real_feature_impact, shap_impact_percent)

    # Store results   
    # Time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Save as List
    lst_result = [
    ('mae_nn', mae_nn),
    ('mae_cobb_nn', mae_cobb_nn),
    ('mse_nn', mse_nn),
    ('mse_cobb_nn', mse_cobb_nn),
    ('cobb_spearman', cobb_spearman),
    ('cobb_cossine', cobb_cossine),
    ('nn_spearman', nn_spearman),
    ('nn_cossine', nn_cossine),
    ('nn_shap_spearman', nn_shap_spearman),
    ('nn_shap_cossine', nn_shap_cossine),
    ('elapsed_time', elapsed_time),
    ('number_of_features', number_of_features),
    ('neural_network_hidden_layers', neural_network_hidden_layers),
    ('activition_function',activition_function),
    ('dropout',dropout),
    ('real_coefs',real_feature_impact),
    ('cobb_coefs',cobb_douglas_feature_impact),
    ('nn_coefs',neural_network_feature_impact)]

    result_list.append(lst_result)
    
# =============================================================================
# 9 - Final Results
# =============================================================================
df, df_cosine_mean, df_spearman_mean, df_mae_mean, df_mse_mean = generate_average_metrics_separated(result_list)
df.to_csv('simulation_result.csv', index=False)

# =============================================================================
# =============================================================================
