# =============================================================================
# Packages
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Cobb-Douglas Activation Function
# =============================================================================
# Activation Function
class CobbDouglasLogActivationFunction(nn.Module):
    def __init__(self, input_dim, output_dim, normalized):
        super().__init__()
        self.alphas = nn.Parameter(torch.rand(output_dim, input_dim) * 0.1)  # (output_dim, input_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))  # (output_dim,)
        self.normalized = normalized

    def forward(self, x):
        """
        x: Tensor de shape (batch_size, input_dim)
        Retorna: Tensor de shape (batch_size, output_dim)
        """
        x = torch.clamp(x, min=1e-6)
        log_x = torch.log(x)  # (batch_size, input_dim)

        if self.normalized == 0:
            alphas = self.alphas  # (output_dim, input_dim)
        else:
            alphas = torch.softmax(self.alphas, dim=1)

        dot = torch.sum(alphas.unsqueeze(0) * log_x.unsqueeze(1), dim=2)

        return torch.exp(dot) + self.bias  # (batch_size, output_dim)
    
# =============================================================================
# =============================================================================