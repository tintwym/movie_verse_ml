# Movie Verse ML

Flask service that classifies movie review text using **BERT** (`bert-base-uncased`) embeddings, **PCA** reduction to 10 dimensions, and an **XGBoost** classifier.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — returns `{"message": "ML API is running!"}` |
| `POST` | `/predict` | Body: `{"review": "<text>"}`. Response: `{"results": 0 or 1}` |

Responses include standard security headers (see `app.py`).

## Run locally

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask torch transformers scikit-learn xgboost joblib numpy
python app.py
```

**Environment variables**

| Variable | Default | Purpose |
|----------|---------|---------|
| `FLASK_HOST` | `0.0.0.0` | Bind address |
| `FLASK_PORT` | `5000` | Port |
| `FLASK_DEBUG` | `false` | Set to `true` only for local debugging |

## Models and data

- `config.py` — model name, paths, target PCA dimensions.
- `xgb_model.pkl` / `pca_model.pkl` — if missing at startup, the code trains placeholder models (see `models/xgb_model.py` and `models/bert_pca.py`). Replace these with real trained artifacts for production.
- `utils/data_processing.py` — text cleaning before embedding.

## Layout

```
movie_verse_ml/
├── app.py                 # Flask routes
├── config.py
├── models/
│   ├── bert_pca.py        # BERT + PCA embedding
│   └── xgb_model.py       # XGBoost load/predict
└── utils/
    ├── data_processing.py
    └── helper.py
```

This folder is part of the larger **Movie Verse** workspace; see the repository root `README.md` for how it fits with the mobile and web clients.
