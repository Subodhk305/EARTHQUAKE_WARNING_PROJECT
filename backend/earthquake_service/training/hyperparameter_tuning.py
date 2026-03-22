# earthquake_service/training/hyperparameter_tuning.py
"""
Hyperparameter optimization using Optuna.
"""
import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

def objective(trial, X_train, y_train):
    """Optuna objective function for XGBoost tuning"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use XGBoost's own CV for faster evaluation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        nfold=5,
        stratified=True,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Return best F1 score
    best_f1 = cv_results['test-f1-mean'].max()
    
    return best_f1

async def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """Run hyperparameter optimization"""
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("Best trial:")
    print(f"  F1 Score: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    
    return study.best_params