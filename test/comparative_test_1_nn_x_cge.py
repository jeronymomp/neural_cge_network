# Reexecutando o código após reset

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim

# Gerando dados sintéticos com 4 setores (camadas)
np.random.seed(42)
X_np = np.linspace(0, 10, 100)
true_params = [0.001, 0.02, 0.001, 0.02, 0.001, 0.02, 0.001, 0.02]  # a1, b1, ..., a4, b4

def true_model(x, params):
    z1 = params[0] + params[1] * x
    z2 = params[2] + params[3] * z1
    z3 = params[4] + params[5] * z2
    y = params[6] + params[7] * z3
    return y

Y_np = true_model(X_np, true_params) + np.random.normal(0, 1.0, size=X_np.shape)

# Modelo CGE-like para estimar os parâmetros
def model_error(params, x, y_real):
    a1, b1, a2, b2, a3, b3, a4, b4 = params
    z1 = a1 + b1 * x
    z2 = a2 + b2 * z1
    z3 = a3 + b3 * z2
    y_pred = a4 + b4 * z3
    return np.mean((y_real - y_pred) ** 2)

initial_guess = [0.0] * 8
result = minimize(model_error, initial_guess, args=(X_np, Y_np))
estimated_params = result.x
print("Estimated parameters (CGE-like):", estimated_params)

# Previsão do modelo CGE
y_pred_cge = true_model(X_np, estimated_params)

# Neural Network com PyTorch
X_torch = torch.tensor(X_np, dtype=torch.float32).view(-1, 1)
Y_torch = torch.tensor(Y_np, dtype=torch.float32).view(-1, 1)

class FourLayerNet(nn.Module):
    def __init__(self):
        super(FourLayerNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 1)
        self.fc4 = nn.Linear(1, 1)

    def forward(self, x):
        z1 = self.fc1(x)
        z2 = self.fc2(z1)
        z3 = self.fc3(z2)
        out = self.fc4(z3)
        return out

model = FourLayerNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Treinamento
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_torch)
    loss = criterion(outputs, Y_torch)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Previsão da rede neural
y_pred_nn = model(X_torch).detach().numpy()

from sklearn.metrics import mean_squared_error, r2_score
# Métricas de avaliação
def evaluate(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

evaluate(Y_np, y_pred_cge, "CGE-like model")
evaluate(Y_np, y_pred_nn, "Neural Network")

# Plot dos resultados
plt.figure(figsize=(10, 5))
plt.scatter(X_np, Y_np, label="True data", color="gray", alpha=0.5)
plt.plot(X_np, y_pred_cge, label="CGE-like prediction", color="blue")
plt.plot(X_np, y_pred_nn, label="NN prediction", color="red", linestyle="--")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.title("Comparison: CGE-like Model vs Neural Network")
plt.legend()
plt.grid(True)
plt.show()



