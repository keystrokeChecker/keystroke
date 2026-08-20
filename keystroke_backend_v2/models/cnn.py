import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LightweightKeystrokeCNN(nn.Module):
    """
    Lightweight 2D CNN for 80x32 Log-Mel Spectrogram keystroke classification.
    Designed for low-latency inference on low-resource machines (Intel N100 / 8GB RAM).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 27, dropout: float = 0.3):
        super(LightweightKeystrokeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, 64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts bottleneck 64-D embedding for Feature Set G."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class PyTorchCNNWrapper:
    """
    Scikit-learn compatible wrapper around LightweightKeystrokeCNN.
    """

    def __init__(self, num_classes: int = 27, epochs: int = 25, batch_size: int = 32, lr: float = 0.001):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightweightKeystrokeCNN(num_classes=num_classes).to(self.device)
        self.classes_ = np.arange(num_classes)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Ensure X is (N, 1, 80, 32)
        if X.ndim == 3:
            X = np.expand_dims(X, 1)
        elif X.ndim == 2:
            # Reshape 1D vector to 2D image (e.g. 80x32 = 2560)
            X = X.reshape(X.shape[0], 1, 80, -1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = np.expand_dims(X, 1)
        elif X.ndim == 2:
            X = X.reshape(X.shape[0], 1, 80, -1)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
