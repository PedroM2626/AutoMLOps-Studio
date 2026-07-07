import logging
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x is of shape (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, kernel_size=2):
        super(TCNForecaster, self).__init__()
        layers = []
        num_channels = [hidden_size] * num_layers
        
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding))
            
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x is (batch_size, sequence_length, input_size)
        # TCN expects (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Take the output of the last time step
        out = self.fc(out[:, :, -1])
        return out


class PyTorchTimeSeriesRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='lstm', hidden_size=64, num_layers=2, epochs=20, lr=1e-3, batch_size=32, seq_len=1, random_state=None):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.random_state = random_state
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.device = 'cpu'

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Deep Learning models are unavailable.")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Scale data for NN stability
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Reshape to (batch, seq_len, features)
        input_size = X_scaled.shape[1]
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.model_type == 'lstm':
            self.model = LSTMForecaster(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif self.model_type == 'tcn':
            self.model = TCNForecaster(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()

        preds_inv = self.scaler_y.inverse_transform(preds).flatten()
        return preds_inv
