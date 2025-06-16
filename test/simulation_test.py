# =============================================================================
# 1 - Packages
# =============================================================================
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
# 2 - Data Generation and Preparation
# =============================================================================
# Simulated data
X, y, coef = make_regression(
    n_samples=1000,
    n_features=5,
    noise=0.1,
    coef=True,  
    random_state=42
)

X = X.astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)

# Actual Impact
# Treinando modelo
linear_model = LinearRegression()
linear_model.fit(X, y)

feature_index = 0

X_perturbed = X.copy()
X_perturbed[:, feature_index] *= 1.01 

y_pred_original = linear_model.predict(X)
y_pred_perturbed = linear_model.predict(X_perturbed)

impact_percent = ((y_pred_perturbed - y_pred_original) / y_pred_original) * 100

mean_impact = np.mean(impact_percent)
print(f"Impacto percentual médio da feature {feature_index}: {mean_impact:.4f}%")


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
epochs = 100

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    def __init__(self,input_dim):
        super().__init__()
        self.cobb_1 = CobbDouglasActivationFunction(input_dim,4)
        self.cobb_2 = CobbDouglasActivationFunction(4,2)
        self.cobb_3 = CobbDouglasActivationFunction(2,1)
    def forward(self, x):
        layer_1_result = self.cobb_1(x)
        layer_2_result = self.cobb_2(layer_1_result)
        layer_3_result = self.cobb_3(layer_2_result)
        return layer_3_result

# 4.3 - Execution
# Initiate Model
cobb_model = CobbDouglasNeuralNetwork(input_size)

# Training
cobb_nn_criterion = nn.MSELoss()
cobb_nn_optimizer = optim.Adam(cobb_model.parameters(), lr=0.001)


for epoch in range(epochs):
    cobb_nn_outputs = cobb_model(X_train_tensor)
    cobb_nn_loss = cobb_nn_criterion(cobb_nn_outputs,y_train_tensor)
    cobb_nn_optimizer.zero_grad()
    cobb_nn_loss.backward()
    cobb_nn_optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {cobb_nn_loss.item():.4f}')
 
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
# 5.0 - Feature under analysis
analyzed_feature = 0

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
cobb_douglas_feature_impact = cobb_douglas_impact(layers, analyzed_feature)
    
#-----------------------------------------------------------------------------#
# Neural Network Impact
x_test_sensitivity_analysis = X_test_tensor.clone()
x_test_sensitivity_analysis[:, 0] *= 1.01

# Predict
base_predicted_nn = model(X_test_tensor)
predicted_nn_sensitivity_analysis = model(x_test_sensitivity_analysis)

diff = torch.abs(predicted_nn_sensitivity_analysis - base_predicted_nn)
mae = diff.mean()
mae.item()

# =============================================================================
# =============================================================================
