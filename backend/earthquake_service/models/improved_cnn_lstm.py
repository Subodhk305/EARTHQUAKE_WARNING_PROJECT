# earthquake_service/models/improved_cnn_lstm.py
"""
Improved CNN-LSTM model with attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNNLSTMModel(nn.Module):
    """Enhanced CNN-LSTM with attention and residual connections"""
    
    def __init__(self, input_channels=3, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        # CNN feature extractor with residual blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Residual block
        self.residual = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256)
        )
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),  # 512 from bidirectional
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Residual connection
        residual = x
        x = self.residual(x)
        x = x + residual
        x = F.relu(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Projection
        features = self.projection(attended)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, features
    
    def extract_features(self, x):
        """Extract features for XGBoost"""
        _, features = self.forward(x)
        return features