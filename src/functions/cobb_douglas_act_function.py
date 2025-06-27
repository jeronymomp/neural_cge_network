# =============================================================================
# Packages
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Cobb-Douglas Activation Function
# =============================================================================
# 4.1 - Activation Function
class CobbDouglasActivationFunction(nn.Module):
    def __init__(self, input_dim, output_dim, normalized):
        super().__init__()
        self.alphas = nn.Parameter(torch.rand(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.normalized = normalized

    def forward(self, x):
        x = torch.clamp(x, min=1e-6)

        if self.normalized == 0:
            x_exp = torch.pow(x.unsqueeze(1), self.alphas)  
            prod = torch.prod(x_exp, dim=2)                 
        else:
            soft_alphas = torch.softmax(self.alphas, dim=1)  
            x_exp = torch.pow(x.unsqueeze(1), soft_alphas)
            prod = torch.prod(x_exp, dim=2)

        return prod + self.bias  
    
# =============================================================================
# =============================================================================
