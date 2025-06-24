# =============================================================================
# 1 - Packages
# =============================================================================
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# Parameters
# =============================================================================
epochs = 1000
learning_rate = 0.01
number_of_features = 3

# =============================================================================
# 2 - Data Generation and Preparation
# =============================================================================
# Simulated data
X, y, coef = make_regression(
    n_samples=1000,
    n_features=number_of_features,
    noise=0.1,
    coef=True,  
    random_state=42
)

X = X.astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)

# Actual Impact
# Training Model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Linear Impact
real_feature_impact = []

for i in range(0, number_of_features):
    X_perturbed = X.copy()
    X_perturbed[:, i] *= 1.01 
    y_pred_original = linear_model.predict(X)
    y_pred_perturbed = linear_model.predict(X_perturbed)
    impact_percent = ((y_pred_perturbed - y_pred_original) / y_pred_original) * 100
    mean_impact = np.mean(impact_percent)
    real_feature_impact.append(mean_impact)

# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensor
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# =============================================================================
# 3 - Basic Neural Network
# =============================================================================
# 3.1 Parameters
input_size = X_train.shape[1] # Input size

# 3.2 Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork,self).__init__()
        self.layer_1 = nn.Linear(input_size,4)
        self.layer_2 = nn.Linear(4,2)
        self.output_layer = nn.Linear(2,1)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x
    
# Intantiate the model
model = NeuralNetwork(input_size)

# 3.3 Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
        
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
print(f'Test Loss: {test_loss.item():.4f}')

# =============================================================================
# 4 - Cobb-Douglas Neural Network
# =============================================================================
# 4.1 - Activation Function
class CobbDouglasActivationFunction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.alphas = nn.Parameter(torch.rand(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        x = torch.clamp(x, min=1e-6)
        x_exp = torch.pow(x.unsqueeze(1), self.alphas)
        prod = torch.prod(x_exp, dim=2)  # (batch_size, output_dim)
        return prod + self.bias

# 4.2 - Cobb Douglas Neural Network
class CobbDouglasNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cobb_1 = CobbDouglasActivationFunction(input_dim, 4)
        self.cobb_2 = CobbDouglasActivationFunction(4, 2)
        self.cobb_3 = CobbDouglasActivationFunction(2, 1)

    def forward(self, x):
        x = self.cobb_1(x)
        x = self.cobb_2(x)
        x = self.cobb_3(x)
        return x

# 4.3 - Training Setup
input_size = X_train_tensor.shape[1]  

cobb_model = CobbDouglasNeuralNetwork(input_size)
cobb_nn_criterion = nn.MSELoss()
cobb_nn_optimizer = optim.Adam(cobb_model.parameters(), lr=learning_rate)

# Gradiente: histórico para análise
gradient_history = {name: [] for name, _ in cobb_model.named_parameters()}

# Loop de treino
for epoch in range(epochs):
    cobb_model.train()
    outputs = cobb_model(X_train_tensor)
    loss = cobb_nn_criterion(outputs, y_train_tensor)

    cobb_nn_optimizer.zero_grad()
    loss.backward()

    # Registrar norma dos gradientes
    for name, param in cobb_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_history[name].append(grad_norm)
        else:
            gradient_history[name].append(0.0)

    cobb_nn_optimizer.step()

    # Mostrar loss
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

medias = {chave: sum(valores)/len(valores) for chave, valores in gradient_history.items()}
 
# =============================================================================
# 5 - Accuracy
# =============================================================================
y_test = y_test_tensor.detach().cpu().numpy()
predicted_nn = model(X_test_tensor).detach().cpu().numpy()
predicted_cobb_nn = cobb_model(X_test_tensor).detach().cpu().numpy()

# Mean Absolute Error
mae_nn = mean_absolute_error(y_test, predicted_nn)
mae_cobb_nn = mean_absolute_error(y_test, predicted_cobb_nn)

# Mean Squared Error
mse_nn = mean_squared_error(y_test, predicted_nn)
mse_cobb_nn = mean_squared_error(y_test, predicted_cobb_nn)

# =============================================================================
# 5 - Sensitivity Analysis
# =============================================================================

#-----------------------------------------------------------------------------#
# 5.1 - Cobb Douglas impact
def cobb_douglas_impact(layers_exponents, feature_index):
    impacto = np.zeros(layers_exponents[0].shape[1])
    impacto[feature_index] = 1.0
    for layer_exp in layers_exponents:
        impacto = layer_exp @ impacto  
    impacto_final = impacto.sum() * 100  # multiplicamos por 100 para percentual
    return impacto_final

# Create array of coef
alphas_1 = cobb_model.cobb_1.alphas.data  
alphas_2 = cobb_model.cobb_2.alphas.data  
alphas_3 = cobb_model.cobb_3.alphas.data  

layer1 = alphas_1.cpu().numpy()
layer2 = alphas_2.cpu().numpy()
layer3 = alphas_3.cpu().numpy()

layers = [layer1, layer2, layer3]

# Impact
cobb_douglas_feature_impact = []
for i in range(number_of_features):
    cd_feature_impact = cobb_douglas_impact(layers, i)
    cobb_douglas_feature_impact.append(cd_feature_impact)

#-----------------------------------------------------------------------------#
# Neural Network Impact
neural_network_feature_impact = []

for i in range(number_of_features):
    x_test_sensitivity_analysis = X_test_tensor.clone()
    x_test_sensitivity_analysis[:, i] *= 1.01

    # Predict
    base_predicted_nn = model(X_test_tensor)
    predicted_nn_sensitivity_analysis = model(x_test_sensitivity_analysis)
    
    diff = torch.abs(predicted_nn_sensitivity_analysis - base_predicted_nn)
    mae = diff.mean()
    neural_network_feature_impact.append(mae.item())

# =============================================================================
# 6 -  SHAP Values
# =============================================================================
def compute_shap_impact_percent(model, X_train_tensor, X_test_tensor, y_train, scaler, 
                               background_size=100, test_sample_size=50, steps=50):
    """
    Compute SHAP values and estimate the approximate percentage impact on the output
    for a 1% variation in each input feature.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model (should be in eval mode).
    X_train_tensor : torch.Tensor
        Training data tensor (standardized).
    X_test_tensor : torch.Tensor
        Test data tensor (standardized).
    y_train : numpy.ndarray
        Training targets (original scale).
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to standardize X_train and X_test.
    background_size : int, optional (default=100)
        Number of samples from training data to use as background for SHAP.
    test_sample_size : int, optional (default=50)
        Number of test samples to explain.
    steps : int, optional (default=50)
        Number of steps for SHAP GradientExplainer (not always needed explicitly).
        
    Returns:
    --------
    impact_percent_relative : np.ndarray
        Array of approximate percentage impacts of 1% input variation per feature.
    """
    model.eval()

    # Select background and test samples
    background = X_train_tensor[:background_size]
    test_samples = X_test_tensor[:test_sample_size]

    # Create SHAP Gradient Explainer
    explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values for test samples
    shap_values = explainer.shap_values(test_samples)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values.squeeze(-1)), axis=0)

    # Undo standardization to get original feature scales
    X_train_original = scaler.inverse_transform(X_train_tensor.cpu().numpy())
    mean_abs_features = np.mean(np.abs(X_train_original), axis=0)

    # Mean output (target) value
    mean_output = np.mean(y_train)

    # Approximate impact for 1% variation (simplifies to mean_abs_shap * 0.01)
    impact_1_percent = mean_abs_shap * 0.01

    # Convert to percentage relative to mean output
    impact_percent_relative = (impact_1_percent / mean_output) * 100

    # Print feature-wise impacts
    for i, imp in enumerate(impact_percent_relative):
        print(f"Feature {i} approximate % impact for 1% input variation: {imp:.4f}%")

    return impact_percent_relative

# Example usage:
impact_percent = compute_shap_impact_percent(model, X_train_tensor, X_test_tensor, y_train, scaler)

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

# Cosine Similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # avoid division by zero
    return dot_product / (norm_vec1 * norm_vec2)

similarity_cobb_douglas = cosine_similarity(real_feature_impact, cobb_douglas_feature_impact)
similarity_neural_network = cosine_similarity(real_feature_impact, neural_network_feature_impact)
similarity_neural_network_shap = cosine_similarity(real_feature_impact, impact_percent)


normalized_cobb_douglas_feature_impact = cobb_douglas_feature_impact / sum(cobb_douglas_feature_impact)
similarity_normalized_cobb_douglas = cosine_similarity(real_feature_impact, normalized_cobb_douglas_feature_impact)

# =============================================================================
# =============================================================================