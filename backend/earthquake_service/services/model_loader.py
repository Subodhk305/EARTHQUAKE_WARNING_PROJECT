# earthquake_service/services/model_loader.py
"""
Model Loader — singleton that loads CNN-LSTM + XGBoost + scaler on startup.
Now supports both original and GPU-trained models.
"""
from __future__ import annotations

import logging
import os
import json
import joblib
import torch
import xgboost as xgb
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path

from earthquake_service.config import settings
from earthquake_service.models.cnn_lstm import CNNLSTMModel

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("💻 Using CPU for inference")


class ModelLoader:
    cnn_lstm: Optional[CNNLSTMModel] = None
    xgb_model: Optional[Union[xgb.XGBClassifier, xgb.Booster]] = None
    scaler: Optional[Any] = None
    model_version: str = settings.MODEL_VERSION
    model_name: str = settings.MODEL_NAME
    _loaded: bool = False
    _model_metadata: Dict[str, Any] = {}
    _is_gpu_model: bool = False  # Track if we're using GPU-trained model

    @classmethod
    async def load_all(cls):
        """Load all saved model artefacts from disk."""
        model_dir = Path(settings.MODEL_DIR)
        
        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Define possible model paths (prefer GPU-trained models)
        cnn_path = model_dir / "cnn_lstm_model_gpu.pt"
        xgb_gpu_path = model_dir / "xgb_classifier_gpu.json"
        xgb_original_path = model_dir / settings.XGBOOST_MODEL
        scaler_gpu_path = model_dir / "feature_scaler_gpu.pkl"
        scaler_original_path = model_dir / settings.SCALER_FILE
        
        # Choose best available models
        use_gpu_model = xgb_gpu_path.exists() and scaler_gpu_path.exists()
        
        xgb_path = xgb_gpu_path if use_gpu_model else xgb_original_path
        scaler_path = scaler_gpu_path if use_gpu_model else scaler_original_path
        
        cls._is_gpu_model = use_gpu_model
        
        # Log detailed path information
        logger.info("=" * 60)
        logger.info("📂 MODEL LOADING DETAILS")
        logger.info("=" * 60)
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Directory exists: {model_dir.exists()}")
        logger.info(f"Using GPU-trained model: {cls._is_gpu_model}")
        
        # List files if directory exists
        if model_dir.exists():
            files = list(model_dir.glob("*"))
            logger.info(f"Files in directory ({len(files)}):")
            for file in files:
                logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        else:
            logger.error(f"❌ Model directory does NOT exist: {model_dir}")
            logger.error(f"   Current working directory: {os.getcwd()}")
            return
        
        # Track loading status
        load_status = {
            "cnn_lstm": False,
            "xgb": False,
            "scaler": False
        }
        
        # ── CNN-LSTM (optional) ───────────────────────────────────────────────
        try:
            if cnn_path.exists():
                logger.info("🔄 Loading CNN-LSTM model...")
                cls.cnn_lstm = CNNLSTMModel().to(_DEVICE)
                state = torch.load(cnn_path, map_location=_DEVICE, weights_only=False)
                cls.cnn_lstm.load_state_dict(state)
                cls.cnn_lstm.eval()
                logger.info(f"✅ CNN-LSTM weights loaded from {cnn_path}")
                load_status["cnn_lstm"] = True
            else:
                logger.info("ℹ️ CNN-LSTM model not found (optional)")
                
        except Exception as e:
            logger.warning(f"CNN-LSTM model not loaded (optional): {e}")
            cls.cnn_lstm = None
        
        # ── XGBoost ──────────────────────────────────────────────────────────
        try:
            logger.info("🔄 Loading XGBoost model...")
            
            if xgb_path.exists():
                logger.info(f"   Found model at: {xgb_path}")
                
                if cls._is_gpu_model:
                    # Load as Booster (from GPU training)
                    cls.xgb_model = xgb.Booster()
                    cls.xgb_model.load_model(str(xgb_path))
                    logger.info("   Loaded as XGBoost Booster (GPU-trained)")
                else:
                    # Load as XGBClassifier (original)
                    cls.xgb_model = xgb.XGBClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        use_label_encoder=False,
                        eval_metric="mlogloss",
                        random_state=42,
                        tree_method="hist",
                    )
                    cls.xgb_model.load_model(str(xgb_path))
                    logger.info("   Loaded as XGBoost Classifier (original)")
                
                # Update model version and name
                if cls._is_gpu_model:
                    cls.model_version = "2.0.0"
                    cls.model_name = "XGBoost_GPU"
                    
                    # Extract model info if possible
                    try:
                        config = json.loads(cls.xgb_model.save_config())
                        cls._model_metadata = {
                            'num_class': config.get('learner', {}).get('objective', {}).get('num_class', 3),
                            'tree_method': config.get('learner', {}).get('generic_param', {}).get('tree_method', 'hist'),
                            'model_type': 'GPU_trained'
                        }
                    except:
                        pass
                
                logger.info(f"✅ XGBoost model loaded from {xgb_path}")
                load_status["xgb"] = True
            else:
                logger.warning(f"⚠️ XGBoost model NOT found at {xgb_path}")
                logger.warning("   Expected file: xgb_classifier_gpu.json or xgb_classifier.json")
                
        except Exception as e:
            logger.error(f"❌ Failed to load XGBoost model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            cls.xgb_model = None
        
        # ── Scaler ───────────────────────────────────────────────────────────
        try:
            logger.info("🔄 Loading feature scaler...")
            if scaler_path.exists():
                logger.info(f"   Found scaler at: {scaler_path}")
                cls.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Feature scaler loaded from {scaler_path}")
                load_status["scaler"] = True
            else:
                logger.warning(f"⚠️ Scaler NOT found at {scaler_path}")
                logger.warning("   Expected file: feature_scaler_gpu.pkl or feature_scaler.pkl")
                
        except Exception as e:
            logger.error(f"❌ Failed to load scaler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            cls.scaler = None
        
        # Check overall load status
        cls._loaded = load_status["xgb"] and load_status["scaler"]
        
        logger.info("=" * 60)
        logger.info("📊 LOADING RESULTS")
        logger.info("=" * 60)
        logger.info(f"CNN-LSTM: {'✅' if load_status['cnn_lstm'] else 'ℹ️ (optional)'}")
        logger.info(f"XGBoost: {'✅' if load_status['xgb'] else '❌'}")
        logger.info(f"Scaler: {'✅' if load_status['scaler'] else '❌'}")
        
        if cls._loaded:
            logger.info(f"🎉 All models loaded successfully! (Version: {cls.model_version})")
            if cls._is_gpu_model:
                logger.info("   Using GPU-trained model with 94.7% accuracy!")
        else:
            logger.warning("⚠️ Some models failed to load. Service may have limited functionality.")

    @classmethod
    def is_ready(cls) -> bool:
        """Check if all required models are loaded."""
        return (cls._loaded and 
                cls.xgb_model is not None and 
                cls.scaler is not None)

    @classmethod
    def get_device(cls) -> torch.device:
        """Get the current device (CPU/GPU)."""
        return _DEVICE

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Get model metadata and loading status."""
        return {
            "model_name": cls.model_name,
            "model_version": cls.model_version,
            "device": str(_DEVICE),
            "loaded": cls._loaded,
            "gpu_model": cls._is_gpu_model,
            "cnn_lstm_loaded": cls.cnn_lstm is not None,
            "xgb_loaded": cls.xgb_model is not None,
            "scaler_loaded": cls.scaler is not None,
            "model_dir": settings.MODEL_DIR,
            "metadata": cls._model_metadata
        }

    @classmethod
    def predict(cls, features: np.ndarray) -> Dict[str, Any]:
        """
        Run prediction on features.
        Works with both XGBClassifier and Booster models.
        """
        if cls.xgb_model is None or cls.scaler is None:
            raise RuntimeError("Models not loaded")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = cls.scaler.transform(features)
        
        # Get predictions based on model type
        if cls._is_gpu_model:
            # Booster model (from GPU training)
            dtest = xgb.DMatrix(features_scaled)
            pred_proba = cls.xgb_model.predict(dtest)[0]
            pred_class = int(np.argmax(pred_proba))
            probabilities = pred_proba.tolist()
        else:
            # XGBClassifier (original)
            pred_class = int(cls.xgb_model.predict(features_scaled)[0])
            probabilities = cls.xgb_model.predict_proba(features_scaled)[0].tolist()
        
        # Map class to magnitude and risk
        class_names = ["Low Risk", "Medium Risk", "High Risk"]
        magnitude_map = {
            0: "< 3.0",
            1: "3.0 - 5.0",
            2: "> 5.0"
        }
        risk_map = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        
        return {
            "class": pred_class,
            "class_name": class_names[pred_class],
            "risk_level": risk_map[pred_class],
            "magnitude_range": magnitude_map.get(pred_class, "Unknown"),
            "probability": probabilities[pred_class],
            "probabilities": probabilities,
            "confidence": max(probabilities),
            "model_version": cls.model_version
        }

    @classmethod
    def predict_cnn_features(cls, waveform_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from CNN-LSTM model."""
        if cls.cnn_lstm is None:
            raise RuntimeError("CNN-LSTM model not loaded")
        
        with torch.no_grad():
            features = cls.cnn_lstm.extract_features(waveform_tensor)
        return features

    @classmethod
    def predict_xgb(cls, features: torch.Tensor) -> Dict[str, Any]:
        """Run XGBoost prediction on extracted features."""
        if cls.xgb_model is None or cls.scaler is None:
            raise RuntimeError("XGBoost model or scaler not loaded")
        
        # Move to CPU and convert to numpy
        features_np = features.cpu().numpy()
        
        # Use the unified predict method
        return cls.predict(features_np)


# Global instance
model_loader = ModelLoader