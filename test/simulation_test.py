import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression



# Gerar dados de exemplo
# Gerar dados com coeficientes explícitos
X, y, coef = make_regression(
    n_samples=1000,
    n_features=5,
    noise=0.1,
    coef=True,  # ← Isso faz retornar os coeficientes verdadeiros
    random_state=42
)

X = X.astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Converter para tensores PyTorch
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork,self).__init__()
        self.layer_1 = nn.Linear(input_size,64)
        self.layer_2 = nn.Linear(64,32)
        self.output_layer = nn.Linear(32,1)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x
    
    
input_size = X_train.shape[1]



model = NeuralNetwork(input_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 100

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









def impacto_individual(model, feature_idx, X_tensor, delta=1.0):
    """
    Calcula: Em média, quanto y muda quando x1 aumenta em 'delta' unidades?
    
    Args:
        model: modelo treinado
        feature_idx: índice da feature (ex: 0 para x1)
        X_tensor: dados de entrada (tensor)
        delta: magnitude da variação
    """
    X_original = X_tensor.clone()
    X_modified = X_tensor.clone()
    
    # Modifica apenas a feature escolhida
    X_modified[:, feature_idx] += delta
    
    with torch.no_grad():
        y_original = model(X_original)
        y_modified = model(X_modified)
    
    diferenca = (y_modified - y_original).mean().item()
    
    print(f"Para cada aumento de {delta} em x{feature_idx+1}, y muda em média {diferenca:.4f}")
    return diferenca / delta  # Retorna "coeficiente" aproximado

# Exemplo de uso:
coef_aprox = impacto_individual(model, feature_idx=0, X_tensor=X_test_tensor)



