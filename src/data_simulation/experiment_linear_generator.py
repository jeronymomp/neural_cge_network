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

# =============================================================================
# 2 - Parameters
# =============================================================================
result_dict = {}

#--------
# 2.1 - Experiment Parameters
sample_size = 10000
number_of_features = 5
random_state = 0
test_sample_size = 0.2
valid_sample_size = 0.2

#--------
# Neural Network Parameters
epochs = 10000
neural_network_hidden_layers = [4,3]
dropout = 0.2
learning_rate = 0.00001
tries_before_stopping = 10

#--------
# Cobb-Douglas Neural Network Parameters
sum_equal_one = 1

#--------
# Pytorch Seed
random.seed(random_state)                          # Python
np.random.seed(random_state)                       # NumPy
torch.manual_seed(random_state)                    # PyTorch CPU
torch.cuda.manual_seed(random_state)               # PyTorch GPU 
torch.cuda.manual_seed_all(random_state)           # Multiple GPUs
torch.backends.cudnn.deterministic = True  # Determinístic
torch.backends.cudnn.benchmark = False     # Disable Non Determinístic

# =============================================================================
# Starting Loop
# =============================================================================
for i in range(0,1):
    # Generate random sample
    random_state = random.randint(0,10000000)
    # Generate Random Number of Features
    number_of_features = random.randint(1,10)
    # Generate Random Number of layers
    number_of_layers = random.randint(1,5)
    # Generate Random Number of Neurons
    neural_network_hidden_layers = [random.randint(0, 100) for _ in range(number_of_layers)]  
    
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
    a_norm = normalize_importance(real_feature_impact)
    b_norm = normalize_importance(cobb_douglas_feature_impact)
    c_norm = normalize_importance(neural_network_feature_impact)
    
    # Cobb Douglas
    cobb_spearman = compare_importance_spearman(real_feature_impact, cobb_douglas_feature_impact)
    cobb_cossine = compare_importance_cosine(real_feature_impact, cobb_douglas_feature_impact)
    
    # Basic Neural Network
    nn_spearman = compare_importance_spearman(real_feature_impact, neural_network_feature_impact)
    nn_cossine = compare_importance_cosine(real_feature_impact, neural_network_feature_impact)
    
    # Basic NN with Shap
    nn_shap_spearman = compare_importance_spearman(real_feature_impact, shap_impact_percent)
    nn_shap_cossine = compare_importance_cosine(real_feature_impact, shap_impact_percent)

# =============================================================================
# 8 - Result Dict
# =============================================================================
    exp_id = 'Exp_' + str(i)
    if exp_id not in result_dict:
        result_dict[exp_id] = []  # inicia uma lista para armazenar múltiplos resultados
    
    # Adiciona o resultado atual à lista
    result_dict[exp_id].append({
        'mae_nn': mae_nn,
        'mae_cobb_nn': mae_cobb_nn,
        'mse_nn': mse_nn,
        'mse_cobb_nn': mse_cobb_nn,
        'cobb_spearman': cobb_spearman,
        'cobb_cossine': cobb_cossine,
        'nn_spearman': nn_spearman,
        'nn_cossine': nn_cossine,
        'nn_shap_spearman': nn_shap_spearman,
        'nn_shap_cossine': nn_shap_cossine
    })

# =============================================================================
# =============================================================================