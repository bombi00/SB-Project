import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, input_dim=29, output_dim=8, dropout=True, dropout_rate=0.3, use_batchnorm=True):
        super(Predictor, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout_enabled = dropout

        # Updated model with more capacity
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) if use_batchnorm else nn.Identity()
        self.fc3 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(dropout_rate) if dropout else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # output layer
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, dim=1)
        return predicted

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
