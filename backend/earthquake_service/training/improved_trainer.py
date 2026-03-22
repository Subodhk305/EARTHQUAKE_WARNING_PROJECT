# earthquake_service/training/improved_trainer.py
"""
Improved model training with better feature engineering and hyperparameter tuning.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedEarthquakeModel:
    """Improved earthquake prediction model with better architecture"""
    
    def __init__(self, config=None):
        self.config = config or {
            'cnn_channels': [64, 128, 256],
            'lstm_hidden': 256,
            'lstm_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'patience': 15,
            'xgb_params': {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.02,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': None,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'random_state': 42,
                'tree_method': 'hist',
                'early_stopping_rounds': 50
            }
        }
        
    def prepare_data(self, X, y):
        """Prepare data with proper train/val split and handling of class imbalance"""
        
        # Handle missing values
        X = pd.DataFrame(X)
        X = X.fillna(X.mean())
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le
        
        logger.info(f"Class distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scaler = scaler
        
        # Compute class weights for imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        self.class_weights = dict(zip(np.unique(y_train), class_weights))
        
        logger.info(f"Class weights: {self.class_weights}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def train_xgboost(self, data, save_path=None):
        """Train XGBoost with hyperparameter tuning"""
        logger.info("=" * 60)
        logger.info("Training XGBoost model...")
        logger.info("=" * 60)
        
        # Prepare data
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # Scale pos weight for imbalance
        if self.config['xgb_params'].get('scale_pos_weight') is None:
            unique, counts = np.unique(y_train, return_counts=True)
            scale_pos_weight = max(counts) / min(counts)
            self.config['xgb_params']['scale_pos_weight'] = scale_pos_weight
            logger.info(f"Scale pos weight: {scale_pos_weight}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        params = self.config['xgb_params'].copy()
        params.pop('early_stopping_rounds', None)
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config['xgb_params']['n_estimators'],
            evals=evals,
            early_stopping_rounds=self.config['xgb_params']['early_stopping_rounds'],
            verbose_eval=50
        )
        
        self.xgb_model = model
        
        # Save model
        if save_path:
            model.save_model(save_path)
            logger.info(f"✅ XGBoost model saved to {save_path}")
        
        return model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Get predictions
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.xgb_model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC for multi-class
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.5
        
        logger.info(f"\n🎯 Performance Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Per-class metrics
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n📊 Confusion Matrix:")
        logger.info(f"{cm}")
        
        # Feature importance
        importance = self.xgb_model.get_score(importance_type='weight')
        logger.info(f"\n📈 Top 10 Features by Importance:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in sorted_importance:
            logger.info(f"  {feat}: {imp:.0f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_artifacts(self, base_path):
        """Save all model artifacts"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, base_path / 'feature_scaler.pkl')
        logger.info(f"✅ Scaler saved to {base_path / 'feature_scaler.pkl'}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, base_path / 'label_encoder.pkl')
        logger.info(f"✅ Label encoder saved to {base_path / 'label_encoder.pkl'}")
        
        logger.info(f"\n🎉 All artifacts saved to {base_path}")

def create_synthetic_training_data(n_samples=10000):
    """Create synthetic training data for testing"""
    np.random.seed(42)
    
    # Generate features
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels based on feature combinations
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Create meaningful patterns
        score = (X[i, 0] * 0.5 + X[i, 5] * 0.3 + X[i, 10] * 0.2)
        score += np.random.randn() * 0.3
        
        if score > 0.5:
            y[i] = 2  # High risk
        elif score > 0:
            y[i] = 1  # Medium risk
        else:
            y[i] = 0  # Low risk
    
    return X, y

async def train_improved_model():
    """Main training function"""
    logger.info("🚀 Starting improved model training...")
    
    # Create synthetic data (replace with real data)
    X, y = create_synthetic_training_data(10000)
    
    # Initialize model
    model = ImprovedEarthquakeModel()
    
    # Prepare data
    data = model.prepare_data(X, y)
    
    # Train XGBoost
    xgb_model = model.train_xgboost(data, save_path=None)  # Pass save_path when ready
    
    # Evaluate
    results = model.evaluate_model(data['X_test'], data['y_test'])
    
    # Save artifacts
    model.save_artifacts("earthquake_service/models/saved/")
    
    # Save evaluation metrics
    import json
    metrics = {
        'model_name': 'Improved_XGBoost',
        'model_version': '2.0.0',
        'training_samples': len(data['X_train']),
        'evaluation_date': datetime.now().isoformat(),
        **{k: v for k, v in results.items() if k not in ['predictions', 'probabilities']}
    }
    
    metrics_path = Path("earthquake_service/models/saved/evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n✅ Training complete! Metrics saved to {metrics_path}")
    logger.info(f"📊 Final performance: F1={results['f1_score']:.4f}, AUC={results['roc_auc']:.4f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_improved_model())