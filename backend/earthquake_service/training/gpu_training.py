# earthquake_service/training/gpu_training.py
"""
GPU-accelerated model training with mixed precision and parallel processing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🚀 Using device: {device}")

if device.type == 'cuda':
    # Enable cuDNN auto-tuner for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    logger.info(f"✅ CUDA optimized: {torch.cuda.get_device_name(0)}")

class GPUTrainer:
    """GPU-accelerated model trainer with mixed precision"""
    
    def __init__(self, model, config=None):
        self.model = model.to(device)
        self.config = config or {
            'batch_size': 128,  # Larger batch size for GPU
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'epochs': 200,
            'patience': 20,
            'mixed_precision': True,
            'gradient_accumulation': 4,
            'num_workers': 4,
            'pin_memory': True
        }
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config['mixed_precision'] and device.type == 'cuda' else None
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2
        )
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()
        
    def create_dataloaders(self, X, y, val_split=0.2):
        """Create GPU-optimized dataloaders"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Clear cache periodically
            if device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return total_loss / len(val_loader), accuracy, f1
    
    def train(self, train_loader, val_loader):
        """Full training loop with early stopping"""
        logger.info("🚀 Starting GPU-accelerated training...")
        
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Logging
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
                logger.info(f"✅ New best model! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        return history

class GPUXGBoostTrainer:
    """XGBoost training with GPU acceleration"""
    
    def __init__(self):
        # Check if GPU is available for XGBoost
        import xgboost as xgb
        self.has_gpu = xgb.__version__ >= '1.0.0' and torch.cuda.is_available()
        
        self.params = {
            'n_estimators': 1000,
            'max_depth': 8,
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
            'tree_method': 'gpu_hist' if self.has_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.has_gpu else 'cpu_predictor',
            'gpu_id': 0 if self.has_gpu else None
        }
        
        if self.has_gpu:
            logger.info("🚀 XGBoost will use GPU acceleration")
        else:
            logger.info("💻 XGBoost will use CPU (install xgboost-gpu for acceleration)")
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with GPU acceleration"""
        
        import xgboost as xgb
        
        # Create DMatrix (GPU version if available)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Move data to GPU if using GPU training
        if self.has_gpu:
            dtrain.set_info(device='cuda')
            dval.set_info(device='cuda')
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        return model

async def train_with_gpu():
    """Main training function with GPU acceleration"""
    
    logger.info("=" * 60)
    logger.info("🚀 Starting GPU-Accelerated Training Pipeline")
    logger.info("=" * 60)
    
    # Generate synthetic data (replace with real data)
    np.random.seed(42)
    n_samples = 50000  # Larger dataset for GPU
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features (use GPU for scaling if possible)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train CNN-LSTM model (if applicable)
    from earthquake_service.models.improved_cnn_lstm import ImprovedCNNLSTMModel
    
    cnn_model = ImprovedCNNLSTMModel(input_channels=n_features, num_classes=3)
    trainer = GPUTrainer(cnn_model)
    
    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(X_train_scaled, y_train)
    
    # Train CNN-LSTM
    history = trainer.train(train_loader, val_loader)
    
    # Extract features
    cnn_model.eval()
    train_features = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            _, features = cnn_model(data)
            train_features.append(features.cpu().numpy())
    train_features = np.vstack(train_features)
    
    # Train XGBoost on extracted features
    xgb_trainer = GPUXGBoostTrainer()
    xgb_model = xgb_trainer.train(train_features, y_train, X_val_scaled, y_val)
    
    # Evaluate
    dtest = xgb.DMatrix(X_test_scaled)
    y_pred_proba = xgb_model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FINAL PERFORMANCE METRICS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Save models
    torch.save(cnn_model.state_dict(), 'earthquake_service/models/saved/cnn_lstm_model_gpu.pt')
    xgb_model.save_model('earthquake_service/models/saved/xgb_classifier_gpu.json')
    joblib.dump(scaler, 'earthquake_service/models/saved/feature_scaler_gpu.pkl')
    
    logger.info("\n✅ Models saved to earthquake_service/models/saved/")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_with_gpu())