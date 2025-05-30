# stress_model.py
import torch
import torch.nn as nn

class SimpleStressModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def load_model(path=None):
    model = StressModel()
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
