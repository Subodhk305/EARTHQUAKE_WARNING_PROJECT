"""
Standalone Evaluation Script.

Generates full evaluation report including:
  - Classification metrics per model
  - Confusion matrix
  - ROC curves
  - Feature importance (XGBoost)
  - Performance comparison table

Usage:
    python -m earthquake_service.training.evaluate \
        --model-dir ./earthquake_service/models/saved \
        --data-dir ./training_data \
        --output-dir ./reports
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
import xgboost as xgb

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["micro", "minor", "moderate", "strong", "major", "great"]


def load_data(data_dir: str):
    waveforms = np.load(os.path.join(data_dir, "waveforms.npy"))
    structured = np.load(os.path.join(data_dir, "structured.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    return waveforms, structured, labels


def extract_cnn_lstm_embeddings(waveforms: np.ndarray, model_dir: str) -> np.ndarray:
    from earthquake_service.models.cnn_lstm import CNNLSTMModel
    from torch.utils.data import TensorDataset, DataLoader

    ckpt = os.path.join(model_dir, "cnn_lstm_model.pt")
    model = CNNLSTMModel().to(DEVICE)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    dataset = TensorDataset(torch.tensor(waveforms, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for (wf,) in loader:
            emb = model.extract_features(wf.to(DEVICE)).cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    plt.figure(figsize=(10, 8))
    sns.set_style("dark")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix — CNN+LSTM+XGBoost", fontsize=14, pad=15)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def plot_roc_curves(y_test: np.ndarray, y_prob: np.ndarray, n_classes: int, output_path: str):
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, color=color, lw=1.5,
                 label=f"{CLASS_NAMES[i]} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — CNN+LSTM+XGBoost")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("ROC curves saved to %s", output_path)


def plot_feature_importance(clf: xgb.XGBClassifier, top_n: int, output_path: str):
    scores = clf.feature_importances_
    idx = np.argsort(scores)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), scores[idx], color="steelblue")
    plt.yticks(range(top_n), [f"feat_{i}" for i in idx])
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Feature importance plot saved to %s", output_path)


def print_comparison_table(results: list[dict]):
    header = f"{'Model':<30} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6}"
    logger.info("\n" + "=" * len(header))
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        logger.info(
            f"{r['model_name']:<30} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} {r['f1_score']:>6.4f} {r['roc_auc']:>6.4f}"
        )
    logger.info("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./earthquake_service/models/saved")
    parser.add_argument("--data-dir", default="./training_data")
    parser.add_argument("--output-dir", default="./reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading data...")
    waveforms, structured, labels = load_data(args.data_dir)

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    _, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=labels)
    w_te, s_te, y_test = waveforms[idx_test], structured[idx_test], labels[idx_test]
    idx_train, _ = train_test_split(
        np.setdiff1d(idx, idx_test), test_size=0.15 / 0.85, random_state=42, stratify=labels[np.setdiff1d(idx, idx_test)]
    )
    w_tr, s_tr, y_train = waveforms[idx_train], structured[idx_train], labels[idx_train]

    logger.info("Extracting CNN-LSTM embeddings...")
    emb_train = extract_cnn_lstm_embeddings(w_tr, args.model_dir)
    emb_test = extract_cnn_lstm_embeddings(w_te, args.model_dir)
    X_train = np.concatenate([emb_train, s_tr], axis=1)
    X_test = np.concatenate([emb_test, s_te], axis=1)

    scaler_path = os.path.join(args.model_dir, "feature_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    n_classes = len(np.unique(labels))
    results = []

    # Baselines
    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=500, n_jobs=-1)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        results.append({
            "model_name": name,
            "accuracy": float((y_pred == y_test).mean()),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")),
        })

    # Our model
    xgb_clf = xgb.XGBClassifier()
    xgb_path = os.path.join(args.model_dir, "xgb_classifier.json")
    if os.path.exists(xgb_path):
        xgb_clf.load_model(xgb_path)
    else:
        logger.warning("XGBoost model not found — skipping hybrid evaluation.")
        return

    y_pred = xgb_clf.predict(X_test)
    y_prob = xgb_clf.predict_proba(X_test)
    results.append({
        "model_name": "CNN+LSTM+XGBoost (Ours)",
        "accuracy": float((y_pred == y_test).mean()),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")),
    })

    print_comparison_table(results)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES[:n_classes], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES[:n_classes], os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc_curves(y_test, y_prob, n_classes, os.path.join(args.output_dir, "roc_curves.png"))
    plot_feature_importance(xgb_clf, top_n=20, output_path=os.path.join(args.output_dir, "feature_importance.png"))

    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump({"models": results, "confusion_matrix": cm.tolist()}, f, indent=2)

    logger.info("✅ Evaluation complete. Reports in %s", args.output_dir)


if __name__ == "__main__":
    main()
