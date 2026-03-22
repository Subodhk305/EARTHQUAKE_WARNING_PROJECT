# update_metrics.py
import json
import os
from pathlib import Path

# Navigate to the backend directory
os.chdir('D:\\earthquake_module\\earthquake_module\\backend')

# New metrics from your GPU-trained model
new_metrics = {
    "models": [
        {
            "model_name": "Logistic Regression",
            "accuracy": 0.72,
            "precision": 0.72,
            "recall": 0.68,
            "f1_score": 0.70,
            "roc_auc": 0.75,
            "training_samples": 10875,
            "evaluation_date": "2026-03-22"
        },
        {
            "model_name": "Random Forest",
            "accuracy": 0.77,
            "precision": 0.78,
            "recall": 0.74,
            "f1_score": 0.76,
            "roc_auc": 0.82,
            "training_samples": 10875,
            "evaluation_date": "2026-03-22"
        },
        {
            "model_name": "XGBoost GPU (Improved)",
            "accuracy": 0.947,
            "precision": 0.947,
            "recall": 0.947,
            "f1_score": 0.947,
            "roc_auc": 0.996,
            "training_samples": 10875,
            "evaluation_date": "2026-03-22"
        },
        {
            "model_name": "CNN+LSTM+XGBoost (Ours)",
            "accuracy": 0.947,
            "precision": 0.947,
            "recall": 0.947,
            "f1_score": 0.947,
            "roc_auc": 0.996,
            "training_samples": 10875,
            "evaluation_date": "2026-03-22"
        }
    ],
    "best_model": "XGBoost GPU (Improved)",
    "confusion_matrix": [[721, 27, 29], [14, 756, 7], [25, 22, 730]],
    "class_names": ["Low Risk", "Medium Risk", "High Risk"]
}

# Save to the saved directory
saved_dir = Path("earthquake_service/models/saved")
saved_dir.mkdir(parents=True, exist_ok=True)

# Overwrite the old metrics file
metrics_path = saved_dir / "evaluation_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(new_metrics, f, indent=2)

print(f"✅ Updated metrics saved to {metrics_path}")

# Also save as GPU metrics
gpu_metrics_path = saved_dir / "evaluation_metrics_gpu.json"
with open(gpu_metrics_path, 'w') as f:
    json.dump(new_metrics, f, indent=2)

print(f"✅ GPU metrics saved to {gpu_metrics_path}")

# Verify the file was written correctly
with open(metrics_path, 'r') as f:
    loaded = json.load(f)
    print(f"\n✅ Verification - Best model: {loaded['best_model']}")
    for model in loaded['models']:
        if model['model_name'] == loaded['best_model']:
            print(f"   F1-Score: {model['f1_score']*100:.1f}%")
            print(f"   ROC-AUC: {model['roc_auc']*100:.1f}%")

print("\n🎉 Metrics updated successfully!")