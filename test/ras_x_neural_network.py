import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from time import time

# 1. Simulação dos dados (input X e target y)
np.random.seed(42)
n_samples = 200
n_features = 4

# Inputs (features)
X = np.random.rand(n_samples, n_features) * 10

# "Pesos verdadeiros" simulados como uma matriz de insumo-produto
true_weights = np.array([
    [0.2, 0.1, 0.4, 0.3],
    [0.1, 0.3, 0.3, 0.3],
    [0.25, 0.25, 0.25, 0.25],
    [0.4, 0.2, 0.2, 0.2]
])

# Alvo simulado: soma ponderada dos inputs por esses "setores"
y = X @ true_weights.mean(axis=0) + np.random.randn(n_samples) * 0.5  # target = média das colunas da matriz * X

# 2. Estimativa com RAS
# Matriz base X0 (pesos aleatórios entre os neurônios)
X0_ras = np.abs(np.random.randn(n_features, n_features))
u = y[:n_features]  # simplificação para um target curto
v = X[:n_features].sum(axis=0)

# Função RAS
def ras(X0, u, v, max_iter=1000, tol=1e-5):
    X = X0.copy().astype(float)
    for _ in range(max_iter):
        row_factors = u / X.sum(axis=1)
        X = (X.T * row_factors).T
        col_factors = v / X.sum(axis=0)
        X = X * col_factors
        if np.allclose(X.sum(axis=1), u, atol=tol) and np.allclose(X.sum(axis=0), v, atol=tol):
            break
    return X

# Aplicando RAS
X_ras = ras(X0_ras, u, v)
weights_ras = X_ras.mean(axis=0)

# Predição com pesos do RAS (regressão linear com pesos fixos)
y_pred_ras = X @ weights_ras
mse_ras = mean_squared_error(y, y_pred_ras)

# 3. Treinamento de rede neural (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(), activation='identity', solver='lbfgs', random_state=42, max_iter=1000)

start = time()
mlp.fit(X, y)
mlp_time = time() - start

y_pred_mlp = mlp.predict(X)
mse_mlp = mean_squared_error(y, y_pred_mlp)

# 4. Comparação
comparison = pd.DataFrame({
    "Método": ["RAS + pesos médios", "Rede Neural (MLP)"],
    "MSE": [mse_ras, mse_mlp],
    "Tempo de Treinamento (s)": [0.0, round(mlp_time, 4)]
})

weights_comparison = pd.DataFrame({
    "Feature": [f"X{i+1}" for i in range(n_features)],
    "Pesos_RAS": weights_ras,
    "Pesos_MLP": mlp.coefs_[0].flatten()
})

comparison, weights_comparison.head()

