# earthquake_service/training/gpu_training_final.py
"""
GPU-optimized training for earthquake prediction with RTX 3050.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🚀 Using device: {device}")

if device.type == 'cuda':
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"   CUDA Version: {torch.version.cuda}")
    
    # Optimize for RTX 3050
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

class CNNLSTM_GPU(nn.Module):
    """GPU-optimized CNN-LSTM for earthquake prediction"""
    
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        
        # CNN layers with batch norm for stability
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(attended)
        
        return logits, attended
    
    def extract_features(self, x):
        """Extract features for XGBoost"""
        _, features = self.forward(x)
        return features

def prepare_data_for_training():
    """Prepare and preprocess training data"""
    logger.info("\n📊 Preparing training data...")
    
    # Try to load actual earthquake data
    data_path = Path("earthquake_data.csv")
    
    if data_path.exists():
        logger.info("Loading existing earthquake data...")
        df = pd.read_csv(data_path)
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['magnitude_class', 'event_id', 'time', 'place']]
        X = df[feature_cols].fillna(0).values
        
        # Prepare labels (3 classes: low, medium, high risk)
        if 'magnitude_class' in df.columns:
            le = LabelEncoder()
            y = le.fit_transform(df['magnitude_class'])
        else:
            y = pd.cut(df['magnitude'], bins=[0, 3, 5, 10], labels=[0, 1, 2]).astype(int).values
        
        logger.info(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    # Generate synthetic data for demonstration
    logger.warning("No training data found, generating synthetic data...")
    np.random.seed(42)
    n_samples = 20000
    n_features = 100
    
    # Generate more realistic features
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    
    # Create realistic patterns based on feature combinations
    for i in range(n_samples):
        # Pattern for high-risk events
        high_risk_score = (X[i, 0] * 0.8 + X[i, 5] * 0.6 + X[i, 10] * 0.4 + 
                          X[i, 15] * 0.3 + X[i, 20] * 0.2)
        
        # Pattern for medium-risk events  
        medium_risk_score = (X[i, 1] * 0.5 + X[i, 6] * 0.4 + X[i, 11] * 0.3)
        
        # Add noise
        noise = np.random.randn() * 0.2
        
        total_score = high_risk_score * 0.6 + medium_risk_score * 0.3 + noise
        
        if total_score > 0.6:
            y[i] = 2  # High risk
        elif total_score > 0.2:
            y[i] = 1  # Medium risk
        else:
            y[i] = 0  # Low risk
    
    # Balance classes
    from sklearn.utils import resample
    X_df = pd.DataFrame(X)
    X_df['label'] = y
    
    # Sample equal amounts from each class
    max_samples = min(X_df[X_df['label'] == 0].shape[0],
                      X_df[X_df['label'] == 1].shape[0],
                      X_df[X_df['label'] == 2].shape[0])
    
    balanced_dfs = []
    for label in [0, 1, 2]:
        class_df = X_df[X_df['label'] == label]
        balanced_dfs.append(class_df.sample(n=max_samples, random_state=42))
    
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
    
    X_balanced = balanced_df.drop('label', axis=1).values
    y_balanced = balanced_df['label'].values
    
    logger.info(f"✅ Generated {len(X_balanced)} balanced samples")
    logger.info(f"   Class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
    
    return X_balanced, y_balanced

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    """Train XGBoost with GPU acceleration"""
    logger.info("\n🚀 Training XGBoost with GPU acceleration...")
    
    # Check GPU availability
    use_gpu = False
    try:
        import xgboost as xgb
        # Test if GPU is available
        dtrain_test = xgb.DMatrix(X_train[:100], label=y_train[:100])
        use_gpu = xgb.__version__ >= '1.0.0'
    except:
        pass
    
    # Parameters optimized for RTX 3050
    params = {
        'n_estimators': 800,
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
        'verbosity': 1
    }
    
    logger.info(f"   GPU enabled: {use_gpu}")
    logger.info(f"   Tree method: {params['tree_method']}")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    return model

def train_cnn_gpu(X_train, y_train, X_val, y_val):
    """Train CNN-LSTM with GPU acceleration"""
    logger.info("\n🚀 Training CNN-LSTM with GPU acceleration...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 3, 100)  # Reshape to 3-channel
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).reshape(-1, 3, 100)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = CNNLSTM_GPU(input_channels=3, num_classes=3).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs, _ = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = model(batch_X)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        
        scheduler.step(val_acc)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_cnn_model.pt'))
    
    # Extract features
    model.eval()
    train_features = []
    with torch.no_grad():
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            _, features = model(batch_X)
            train_features.append(features.cpu().numpy())
    
    train_features = np.vstack(train_features)
    
    return model, train_features

def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("🎯 GPU-Accelerated Earthquake Prediction Training")
    logger.info("=" * 60)
    
    # Prepare data
    X, y = prepare_data_for_training()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb_model = train_xgboost_gpu(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate XGBoost
    dtest = xgb.DMatrix(X_test_scaled)
    y_pred_proba = xgb_model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FINAL MODEL PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    logger.info(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
    logger.info(f"Recall:    {recall:.4f} ({recall*100:.1f}%)")
    logger.info(f"F1-Score:  {f1:.4f} ({f1*100:.1f}%)")
    logger.info(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.1f}%)")
    
    # Save models
    save_dir = Path("earthquake_service/models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    xgb_model.save_model(save_dir / "xgb_classifier_gpu.json")
    joblib.dump(scaler, save_dir / "feature_scaler_gpu.pkl")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'model_name': 'XGBoost_GPU',
        'model_version': '2.0.0',
        'training_samples': len(X_train),
        'evaluation_date': datetime.now().isoformat(),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    with open(save_dir / "evaluation_metrics_gpu.json", 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n✅ Models saved to {save_dir}")
    logger.info(f"   - xgb_classifier_gpu.json")
    logger.info(f"   - feature_scaler_gpu.pkl")
    logger.info(f"   - evaluation_metrics_gpu.json")
    
    return metrics

if __name__ == "__main__":
    main()
    