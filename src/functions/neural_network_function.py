# =============================================================================
# Packages
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Flexible Neural Network
# =============================================================================
# =============================================================================
# Packages
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Flexible Neural Network
# =============================================================================
class FlexibleNeuralNetwork:
    def __init__(self, X_train, X_valid, X_test, y_train, y_valid, y_test, device = None):
        # Call GPU if available
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
        self.y_train_tensor = torch.from_numpy(y_train).float().to(self.device)
        self.X_valid_tensor = torch.from_numpy(X_valid).float().to(self.device)
        self.y_valid_tensor = torch.from_numpy(y_valid).float().to(self.device)
        self.X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
        self.y_test_tensor = torch.from_numpy(y_test).float().to(self.device)

    def train_model(
        self,
        hidden_layers=[64, 32],
        activation_fn=nn.ReLU,
        loss_fn=nn.MSELoss,
        optimizer_class=optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        dropout_rate=0.0,
        use_batchnorm=False,
        early_stopping_patience=20,
        epochs=200,
        print_every=10,
        verbose=True
    ):
        input_size = self.X_train_tensor.shape[1]
        output_size = self.y_train_tensor.shape[1]

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                layers = []
                sizes = [input_size] + hidden_layers
                for i in range(len(sizes) - 1):
                    layers.append(nn.Linear(sizes[i], sizes[i+1]))
                    if use_batchnorm:
                        layers.append(nn.BatchNorm1d(sizes[i+1]))
                    layers.append(activation_fn())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.Linear(sizes[-1], output_size))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        model = Net().to(self.device)
        criterion = loss_fn()
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            outputs = model(self.X_train_tensor)
            train_loss = criterion(outputs, self.y_train_tensor)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_valid_tensor)
                val_loss = criterion(val_outputs, self.y_valid_tensor)

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Avaliação no teste
        model.eval()
        with torch.no_grad():
            test_outputs = model(self.X_test_tensor)
            test_loss = criterion(test_outputs, self.y_test_tensor)

        if verbose:
            print(f"Test Loss: {test_loss.item():.4f}")

        # Construir descrição da arquitetura da rede
        architecture = []
        for layer in model.model:
            architecture.append(str(layer))

        # Obter pesos (state_dict)
        weights = model.state_dict()

        return model, train_losses, val_losses, best_val_loss, test_loss.item(), architecture, weights


# =============================================================================
# =============================================================================
