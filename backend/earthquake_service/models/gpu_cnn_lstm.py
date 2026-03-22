# earthquake_service/models/gpu_cnn_lstm.py
"""
GPU-optimized CNN-LSTM model with efficient memory usage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPUCNNLSTMModel(nn.Module):
    """GPU-optimized CNN-LSTM model with efficient architecture"""
    
    def __init__(self, input_channels=3, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        # Optimized CNN layers for GPU
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            )
        ])
        
        # Optimized LSTM with cuDNN optimizations
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Efficient attention
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(attended)
        
        return logits, attended
    
    def extract_features(self, x):
        """Extract features for XGBoost"""
        _, features = self.forward(x)
        return features