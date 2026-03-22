# SeismoAI — Earthquake Prediction Module
### AI-Powered Global Disaster Forecasting and Resource Allocation System
**BTech Final Year Project — Department of Computer Engineering, PCCOE Pune**

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │          CLIENT (React + Mapbox GL)          │
                        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
                        │  │ Map View │  │ Pred Panel│  │ Metrics  │  │
                        │  └────┬─────┘  └────┬──────┘  └────┬─────┘  │
                        └───────┼─────────────┼──────────────┼────────┘
                                │  REST API   │  WebSocket   │
                        ┌───────▼─────────────▼──────────────▼────────┐
                        │             FastAPI Backend                   │
                        │  /predict  /historical  /model-metrics  /ws  │
                        └──────────┬──────────────────┬────────────────┘
                                   │                  │
              ┌────────────────────▼──┐    ┌─────────▼──────────────────┐
              │   Prediction Service  │    │     Data Services           │
              │  ┌──────────────────┐ │    │  ┌──────────────────────┐  │
              │  │  IRIS Fetcher    │ │    │  │  USGS Historical DB  │  │
              │  │  (ObsPy/FDSN)    │ │    │  │  (PostgreSQL)        │  │
              │  └────────┬─────────┘ │    │  └──────────────────────┘  │
              │  ┌────────▼─────────┐ │    │  ┌──────────────────────┐  │
              │  │  Feature Eng.   │ │    │  │  Redis Cache         │  │
              │  └────────┬─────────┘ │    │  └──────────────────────┘  │
              │  ┌────────▼─────────┐ │    └────────────────────────────┘
              │  │  CNN-LSTM Model  │ │
              │  │  (PyTorch)       │ │
              │  └────────┬─────────┘ │
              │  ┌────────▼─────────┐ │
              │  │  XGBoost Classi. │ │
              │  └──────────────────┘ │
              └───────────────────────┘
```

### Prediction Pipeline

```
IRIS Waveform (3-ch, 600s)
        │
┌───────▼──────────────────────────────────────────────┐
│  Pre-processing                                        │
│  detrend → taper → bandpass(1-40Hz) → resample(100Hz) │
└───────────────────────────────────┬──────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────────┐
              │  1D CNN (3 blocks: 32→64→128 channels)          │
              │  ConvBlock = Conv1d → BatchNorm → ReLU → MaxPool │
              └─────────────────────┬──────────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────────┐
              │  Bidirectional LSTM (2 layers, 256 hidden)      │
              └─────────────────────┬──────────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────────┐
              │  Soft Attention Pooling                         │
              └─────────────────────┬──────────────────────────┘
                                    │ embedding (128-dim)
                                    ├───────────────────────────────────┐
              USGS Historical       │                                   │
              Seismicity (30d) ─────► Feature Engineering               │
              Geographic Proxy ─────► (25 seismic + 10 geo = 35-dim)   │
                                    │                                   │
                                    ▼                                   │
                               concat(128 + 35) = 163-dim ─────────────┘
                                    │
              ┌─────────────────────▼──────────────────────────┐
              │  XGBoost Classifier (500 trees)                 │
              │  Output: P(micro|minor|moderate|strong|major|   │
              │              great) per magnitude class         │
              └─────────────────────┬──────────────────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  Risk Level Mapping     │
                        │  P(strong+major+great)  │
                        │  <0.40 → Low            │
                        │  0.40-0.70 → Medium     │
                        │  >0.70 → High           │
                        └────────────────────────┘
```

---

## Folder Structure

```
earthquake_module/
├── docker-compose.yml
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   └── earthquake_service/
│       ├── main.py                    # FastAPI application
│       ├── config.py                  # Settings via pydantic-settings
│       │
│       ├── api/
│       │   └── routes.py              # POST /predict, GET /historical, GET /model-metrics
│       │
│       ├── models/
│       │   ├── cnn_lstm.py            # CNN-LSTM PyTorch architecture
│       │   ├── db_models.py           # SQLAlchemy ORM models
│       │   ├── schemas.py             # Pydantic request/response schemas
│       │   └── saved/                 # Trained weights (generated by training)
│       │
│       ├── services/
│       │   ├── iris_fetcher.py        # IRIS FDSN waveform download + preprocessing
│       │   ├── feature_engineering.py # Feature extraction (seismicity + geographic)
│       │   ├── model_loader.py        # Model singleton loader
│       │   └── predictor.py           # End-to-end inference pipeline
│       │
│       ├── training/
│       │   ├── ingest_data.py         # Download USGS catalog → PostgreSQL
│       │   ├── prepare_training_data.py # DB records → .npy training arrays
│       │   ├── train.py               # Full training: CNN-LSTM + XGBoost
│       │   └── evaluate.py            # Evaluation + metrics/plots
│       │
│       ├── utils/
│       │   ├── database.py            # Async SQLAlchemy engine
│       │   ├── cache.py               # Redis async wrapper
│       │   └── logger.py              # Structured logging
│       │
│       └── websocket/
│           └── manager.py             # WebSocket alert broadcaster
│
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.jsx                   # React entry point
        ├── styles/globals.css         # Futuristic dark theme CSS
        ├── services/api.js            # Axios + WebSocket client
        ├── components/
        │   ├── EarthquakeMap.jsx      # Mapbox GL globe + heatmap
        │   ├── PredictionPanel.jsx    # Prediction results display
        │   ├── ProbabilityGauge.jsx   # SVG arc gauge (animated)
        │   ├── ConfidenceMeter.jsx    # Animated confidence bar
        │   ├── AlertToast.jsx         # Real-time alert notifications
        │   ├── LocationSearch.jsx     # Mapbox geocoding search
        │   └── MetricsChart.jsx       # Recharts model comparison
        └── pages/
            └── Dashboard.jsx          # Main dashboard page
```

---

## Setup & Deployment Instructions

### Prerequisites
- Docker + Docker Compose
- Node.js 20+ (for local frontend development)
- Python 3.11+ (for local backend development)
- Mapbox API token (free tier works)
- PostgreSQL 16 (via Docker)

### 1. Environment Setup

```bash
# Backend
cp backend/.env.example backend/.env
# Edit backend/.env and set:
#   DATABASE_URL, REDIS_URL (already set for Docker)

# Frontend — create .env
echo "VITE_MAPBOX_TOKEN=pk.your_real_token_here" > frontend/.env
echo "VITE_API_URL=http://localhost:8000/api/v1" >> frontend/.env
echo "VITE_WS_URL=ws://localhost:8000" >> frontend/.env
```

### 2. Docker Compose (Production-like)

```bash
# Build and start all services
docker-compose up --build

# Access:
#   Frontend:  http://localhost:3000
#   API Docs:  http://localhost:8000/docs
#   WebSocket: ws://localhost:8000/alerts
```

### 3. Training Pipeline

**Step 1: Ingest USGS historical data**
```bash
cd backend
pip install -r requirements.txt
python -m earthquake_service.training.ingest_data \
    --start-year 2000 \
    --end-year 2023 \
    --min-magnitude 2.0
# This downloads ~200k+ events from USGS FDSN API
```

**Step 2: Prepare training arrays**
```bash
python -m earthquake_service.training.prepare_training_data \
    --output-dir ./training_data \
    --max-events 10000
# WARNING: This fetches IRIS waveforms per event — slow!
# For fast testing, skip this step; train.py generates synthetic data
```

**Step 3: Train models**
```bash
python -m earthquake_service.training.train \
    --data-dir ./training_data \
    --model-dir ./earthquake_service/models/saved \
    --epochs 30 \
    --batch-size 16
# Outputs: cnn_lstm_model.pt, xgb_classifier.json, feature_scaler.pkl
```

**Step 4: Evaluate**
```bash
python -m earthquake_service.training.evaluate \
    --model-dir ./earthquake_service/models/saved \
    --data-dir ./training_data \
    --output-dir ./reports
# Outputs: confusion_matrix.png, roc_curves.png, evaluation_results.json
```

### 4. Local Development

```bash
# Backend
cd backend
uvicorn earthquake_service.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

---

## How IRIS Waveform Fetching Works

```python
# The fetcher uses ObsPy's FDSN client:
from obspy.clients.fdsn import Client

client = Client("IRIS")

# 1. Find nearest broadband stations within 5° of target
inventory = client.get_stations(
    latitude=lat, longitude=lon, maxradius=5.0,
    channel="HH?,BH?", level="channel"
)

# 2. Download 10 minutes of waveform data
stream = client.get_waveforms(
    network, station, "*", "HH?,BH?", t_start, t_end
)

# 3. Pre-process pipeline:
stream.detrend("demean")
stream.detrend("linear")
stream.taper(0.05)
stream.filter("bandpass", freqmin=1.0, freqmax=40.0, zerophase=True)
stream.resample(100)

# 4. Convert to numpy array: (3, 60000) — Z, N, E channels
```

Results are cached in Redis for 5 minutes to avoid redundant IRIS requests.

---

## API Reference

### POST /api/v1/predict

**Request:**
```json
{
  "latitude": 35.6762,
  "longitude": 139.6503,
  "location_name": "Tokyo, Japan",
  "radius_km": 200,
  "include_waveform": true
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4-...",
  "location": "Tokyo, Japan",
  "latitude": 35.6762,
  "longitude": 139.6503,
  "probability": 0.7823,
  "predicted_magnitude_class": "strong",
  "risk_level": "High",
  "confidence": 0.8341,
  "magnitude_estimate": "5.0–5.9",
  "nearby_active_faults": 14,
  "recent_seismicity_score": 0.67,
  "model_version": "1.0.0",
  "processing_time_ms": 1243.5,
  "timestamp": "2024-02-18T10:30:00Z"
}
```

### GET /api/v1/historical/{location}
Query params: `lat`, `lon`, `radius_km`, `days`

### GET /api/v1/model-metrics
Returns full evaluation table for all models.

### WebSocket /alerts
Connect for real-time High Risk event broadcasts.

---

## Performance Targets

| Metric | Target | Architecture |
|--------|--------|-------------|
| API response | < 2s | FastAPI + Redis caching |
| Waveform fetch | < 5s | Async IRIS + 5min cache |
| WebSocket latency | < 1s | Native WebSocket |
| Concurrent users | 5000+ | Async + 4 uvicorn workers |
| F1-Score | ≥ 0.85 | CNN+LSTM+XGBoost |
| ROC-AUC | ≥ 0.90 | Hybrid model |
| M>6.0 Precision | ≥ 0.92 | Class-weighted training |

---

## Model Performance (Expected)

| Model | Precision | Recall | F1 | AUC |
|-------|-----------|--------|-----|-----|
| Logistic Regression | 0.72 | 0.68 | 0.70 | 0.75 |
| Random Forest | 0.78 | 0.74 | 0.76 | 0.82 |
| LSTM Only | 0.83 | 0.79 | 0.81 | 0.87 |
| **CNN+LSTM+XGBoost** | **0.89** | **0.86** | **0.87** | **0.93** |

---

## UI Features

- **Black futuristic theme** (#0B0F19 background, neon blue/purple accents)
- **Interactive Mapbox GL globe** with heatmap layer
- **Animated SVG probability gauge** (arc animation, neon glow)
- **Animated confidence meter** with transition
- **Real-time alert toasts** via WebSocket (slide-in animation, risk color-coded)
- **Location search** via Mapbox Geocoding API
- **Model metrics dashboard** with Recharts bar chart + comparison table
- **Historical earthquake overlay** with magnitude-coded circle layer
- **Glassmorphism panels** with backdrop blur

---

## Scalability Notes

- **4 Uvicorn workers** handle concurrent requests
- **Redis caching** reduces IRIS/USGS API calls
- **Async database** (asyncpg) for non-blocking I/O
- **WebSocket ConnectionManager** supports 1000s of concurrent alert subscribers
- **Docker Compose** ready for Kubernetes migration
- **Model inference runs on CPU** by default; set CUDA for GPU acceleration

---

## Data Sources

| Source | Data Type | API |
|--------|-----------|-----|
| USGS Earthquake Catalog | Historical events (M≥2.0, 2000-2023) | `earthquake.usgs.gov/fdsnws` |
| IRIS FDSN | Real-time seismic waveforms (Z,N,E channels) | `service.iris.edu/fdsnws` |
| NOAA GHCN | Meteorological auxiliary features (optional) | `www.ncdc.noaa.gov` |
| EM-DAT | Disaster validation records | `www.emdat.be` |
