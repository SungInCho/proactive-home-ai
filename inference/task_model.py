import torch
import torch.nn as nn

# Simple Multi-Layer Perceptron
class TaskInferenceModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim), # 67 -> 128
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # 128 -> 64
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes) # 64 -> 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        return self.classifier(x)
    
