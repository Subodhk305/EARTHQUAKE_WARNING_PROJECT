"""
Full Training Pipeline.

Step 1: Pre-train CNN-LSTM on synthetic/real waveform data to learn
        seismic embeddings (supervised with magnitude class labels).
Step 2: Extract embeddings for all training samples.
Step 3: Combine embeddings with structured features.
Step 4: Train XGBoost classifier on the combined feature vectors.
Step 5: Save models, scaler, and evaluation metrics.

Usage:
    python -m earthquake_service.training.train \
        --data-dir ./training_data \
        --model-dir ./earthquake_service/models/saved \
        --epochs 30 \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Optimize for GPU training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info("Training device: %s (%s)", DEVICE, torch.cuda.get_device_name(0))
else:
    logger.info("Training device: %s", DEVICE)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeismicDataset(Dataset):
    """
    Loads pre-processed waveform numpy arrays + structured features.
    Expected files in data_dir:
        waveforms.npy       — shape (N, 3, 60000)  float32
        structured.npy      — shape (N, 35)         float32  (seismicity + geo)
        labels.npy          — shape (N,)            int32    (magnitude class 0-5)
    """

    def __init__(self, waveforms, structured, labels):
        self.waveforms = torch.tensor(waveforms, dtype=torch.float32)
        self.structured = torch.tensor(structured, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.waveforms[idx], self.structured[idx], self.labels[idx]


def load_or_create_dummy_data(data_dir: str, n_samples: int = 2000):
    """
    Load real .npy files if available, otherwise create synthetic data
    for pipeline validation.  Replace with real data for production.
    """
    w_path = os.path.join(data_dir, "waveforms.npy")
    s_path = os.path.join(data_dir, "structured.npy")
    l_path = os.path.join(data_dir, "labels.npy")

    if all(os.path.exists(p) for p in [w_path, s_path, l_path]):
        logger.info("Loading training data from %s", data_dir)
        waveforms = np.load(w_path)
        structured = np.load(s_path)
        labels = np.load(l_path)
    else:
        logger.warning("Training data not found — generating SYNTHETIC data for validation.")
        logger.warning("Run ingest_data.py + prepare_training_data.py to use real data.")
        np.random.seed(42)
        waveforms = np.random.randn(n_samples, 3, 60_000).astype(np.float32)
        structured = np.random.randn(n_samples, 35).astype(np.float32)
        # Class distribution: 5 classes (0-4) for magnitude classes
        # micro, minor, moderate, strong, major
        labels = np.random.choice([0, 1, 2, 3, 4],
                                  size=n_samples,
                                  p=[0.10, 0.40, 0.30, 0.15, 0.05]).astype(np.int32)

    return waveforms, structured, labels


# ── Phase 1: CNN-LSTM Pre-training ────────────────────────────────────────────

def pretrain_cnn_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_dir: str,
    epochs: int,
) -> nn.Module:
    from earthquake_service.models.cnn_lstm import CNNLSTMModel
    model = CNNLSTMModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # FIX: Use standard CrossEntropyLoss without weights for now
    # This avoids the class count mismatch issue
    criterion = nn.CrossEntropyLoss()
    
    # Alternative if you want class weights (uncomment after verifying class count):
    # all_labels = np.concatenate([y.numpy() for _, _, y in train_loader])
    # n_classes = len(np.unique(all_labels))
    # class_weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    # weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(model_dir, "cnn_lstm_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for wf, _, labels in train_loader:
            # Move data to device
            wf, labels = wf.to(DEVICE), labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                logits, _ = model(wf)
                loss = criterion(logits, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for wf, _, labels in val_loader:
                wf, labels = wf.to(DEVICE), labels.to(DEVICE)
                
                # No need for autocast during validation
                logits, _ = model(wf)
                val_loss += criterion(logits, labels).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  ✔ Best model saved (val_loss=%.4f)", val_loss)

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    logger.info("CNN-LSTM pre-training complete. Best val_loss=%.4f", best_val_loss)
    return model


# ── Phase 2: Extract embeddings ───────────────────────────────────────────────

def extract_embeddings(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, structured_all, labels_all = [], [], []
    model.eval()
    with torch.no_grad():
        for wf, struct, labels in loader:
            wf = wf.to(DEVICE)
            # Use autocast for faster embedding extraction on GPU
            with autocast():
                emb = model.extract_features(wf).cpu().numpy()
            embeddings.append(emb)
            structured_all.append(struct.numpy())
            labels_all.append(labels.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    structured_all = np.concatenate(structured_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    feat_matrix = np.concatenate([embeddings, structured_all], axis=1)
    return feat_matrix, labels_all, embeddings


# ── Phase 3: XGBoost training ─────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    model_dir: str,
) -> xgb.XGBClassifier:
    n_classes = len(np.unique(y_train))
    
    # Enable GPU for XGBoost if available
    tree_method = "gpu_hist" if torch.cuda.is_available() else "hist"
    
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        tree_method=tree_method,  # Use GPU if available
        predictor="gpu_predictor" if torch.cuda.is_available() else "cpu_predictor",
        n_jobs=-1,
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    
    xgb_path = os.path.join(model_dir, "xgb_classifier.json")
    clf.save_model(xgb_path)
    logger.info("XGBoost model saved to %s (using %s)", xgb_path, tree_method)
    return clf


# ── Phase 4: Evaluation ───────────────────────────────────────────────────────

def evaluate_all(
    X_train, X_test, y_train, y_test,
    cnn_lstm_model, model_dir,
):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    results = []
    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }

    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        results.append({
            "model_name": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")),
            "training_samples": int(len(y_train)),
            "evaluation_date": time.strftime("%Y-%m-%d"),
        })
        logger.info("[%s] F1=%.4f, AUC=%.4f", name, results[-1]["f1_score"], results[-1]["roc_auc"])

    # Load and evaluate the trained XGBoost
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(os.path.join(model_dir, "xgb_classifier.json"))
    y_pred = xgb_clf.predict(X_test)
    y_prob = xgb_clf.predict_proba(X_test)
    results.append({
        "model_name": "CNN+LSTM+XGBoost (Ours)",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")),
        "training_samples": int(len(y_train)),
        "evaluation_date": time.strftime("%Y-%m-%d"),
    })

    # Update class names to match 5 classes
    MAG_CLASS_LABELS = ["micro", "minor", "moderate", "strong", "major"]
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics_payload = {
        "models": results,
        "best_model": "CNN+LSTM+XGBoost (Ours)",
        "confusion_matrix": cm,
        "class_names": MAG_CLASS_LABELS,
    }
    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Evaluation metrics saved to %s", metrics_path)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=MAG_CLASS_LABELS, zero_division=0))
    return metrics_payload


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./training_data")
    parser.add_argument("--model-dir", default="./earthquake_service/models/saved")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-cnn-training", action="store_true",
                        help="Skip CNN-LSTM training if weights already exist.")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    logger.info("=== PHASE 0: Loading data ===")
    waveforms, structured, labels = load_or_create_dummy_data(args.data_dir)
    logger.info("Dataset: %d samples, waveforms=%s, structured=%s", len(labels), waveforms.shape, structured.shape)

    # Train/val/test split (70/15/15)
    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=labels)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15 / 0.85, random_state=42, stratify=labels[idx_tv])

    def subset(idx):
        return waveforms[idx], structured[idx], labels[idx]

    w_tr, s_tr, l_tr = subset(idx_train)
    w_va, s_va, l_va = subset(idx_val)
    w_te, s_te, l_te = subset(idx_test)

    train_ds = SeismicDataset(w_tr, s_tr, l_tr)
    val_ds = SeismicDataset(w_va, s_va, l_va)
    test_ds = SeismicDataset(w_te, s_te, l_te)
    full_ds = SeismicDataset(waveforms, structured, labels)

    # Increase num_workers for faster data loading on GPU systems
    num_workers = 4 if torch.cuda.is_available() else 2
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    logger.info("=== PHASE 1: CNN-LSTM Pre-training ===")
    ckpt = os.path.join(args.model_dir, "cnn_lstm_model.pt")
    if args.skip_cnn_training and os.path.exists(ckpt):
        from earthquake_service.models.cnn_lstm import CNNLSTMModel
        model = CNNLSTMModel().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        logger.info("Skipping CNN-LSTM training, using existing checkpoint.")
    else:
        model = pretrain_cnn_lstm(train_loader, val_loader, args.model_dir, args.epochs)

    logger.info("=== PHASE 2: Embedding Extraction ===")
    X_train_emb, y_train, _ = extract_embeddings(model, DataLoader(SeismicDataset(w_tr, s_tr, l_tr), 
                                                                   batch_size=args.batch_size * 2, 
                                                                   shuffle=False, 
                                                                   num_workers=num_workers,
                                                                   pin_memory=torch.cuda.is_available()))
    X_val_emb, y_val, _ = extract_embeddings(model, DataLoader(SeismicDataset(w_va, s_va, l_va), 
                                                               batch_size=args.batch_size * 2, 
                                                               shuffle=False,
                                                               num_workers=num_workers,
                                                               pin_memory=torch.cuda.is_available()))
    X_test_emb, y_test, _ = extract_embeddings(model, DataLoader(SeismicDataset(w_te, s_te, l_te), 
                                                                 batch_size=args.batch_size * 2, 
                                                                 shuffle=False,
                                                                 num_workers=num_workers,
                                                                 pin_memory=torch.cuda.is_available()))

    logger.info("=== PHASE 2b: Feature scaling ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_val_scaled = scaler.transform(X_val_emb)
    X_test_scaled = scaler.transform(X_test_emb)
    joblib.dump(scaler, os.path.join(args.model_dir, "feature_scaler.pkl"))
    logger.info("Scaler saved.")

    logger.info("=== PHASE 3: XGBoost Training ===")
    train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, args.model_dir)

    logger.info("=== PHASE 4: Evaluation ===")
    evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test, model, args.model_dir)

    logger.info("🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()