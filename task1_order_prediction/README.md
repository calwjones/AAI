# Task 1 — Intelligent Order Prediction Service

FastAPI service that analyses a customer's purchase history and returns the products they're most likely to re-order. Built to the case study spec: simple ML algorithm acceptable, must expose a `/predict` endpoint.

**Port:** 8002
**Model:** `RandomForestClassifier` (scikit-learn)

## How it works

On startup, the service loads `data/Order_history.csv` and builds features for every customer–product pair the customer has bought before. Given a `customer_id`, it scores all those pairs and returns the top-N by predicted re-order probability.

### Features (per customer–product pair)

| Feature | Meaning |
|---|---|
| `order_count` | Times the customer bought this product |
| `days_since_last_order` | Days since the last purchase |
| `avg_quantity` | Average quantity per order |
| `order_gap_std` | Standard deviation of intervals between orders (regularity signal) |
| `total_spend` | Total money spent on this product |
| `customer_total_orders` | Total unique orders by this customer |
| `product_popularity` | Unique customers who bought this product |

### Model

- `RandomForestClassifier`, 400 trees, `max_depth=16`, `class_weight="balanced"`
- 80/20 temporal split on order date (cutoff 2026-01-13)
- Binary target: did the customer–product pair reappear after the cutoff
- Test performance: accuracy 99.2%, ROC-AUC 0.998

Metadata in `models/model_metadata.json`. Trained model checked into `models/order_prediction_model.pkl`.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status and whether the model is loaded |
| `POST` | `/predict` | Body `{customer_id, top_n}` → ranked recommendations |
| `GET` | `/customers` | All known customer IDs |
| `GET` | `/metadata` | Model hyperparameters, features, test metrics |

Example:

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 101, "top_n": 5}'
```

Response:

```json
{
  "customer_id": 101,
  "recommendations": [
    {"product_id": 42, "product_name": "...", "reorder_probability": 0.94}
  ]
}
```

Interactive docs at http://localhost:8002/docs.

## Audit logging

Every successful `/predict` call is logged to DESD via the same `InteractionLogger` the quality service uses. Payload sent to `/api/ai-logs/` includes `service_type: "order_prediction"`, the `customer_id` and `top_n` from the request, the ranked recommendations, the top recommendation's probability as `confidence_score`, and the model version. DESD credentials come from the `DESD_SERVICE_USERNAME` / `DESD_SERVICE_PASSWORD` environment variables.

## Running

Via docker-compose (from repo root):

```bash
docker compose up ai-demand-service --build
```

Standalone (Python 3.11):

```bash
pip install -r requirements.txt
cd src && uvicorn service:app --port 8002 --reload
```

## Tests

```bash
pytest tests/
```

Seven tests: health, predict on known customer, `top_n` enforcement, descending sort on probability, 404 on unknown customer, `/customers`, `/metadata`.

## Layout

```
task1_order_prediction/
├── src/
│   ├── service.py        # FastAPI app
│   └── prediction.py     # OrderPredictor: features + inference
├── models/
│   ├── order_prediction_model.pkl
│   └── model_metadata.json
├── data/
│   └── Order_history.csv
├── notebooks/
│   └── 01_training.ipynb
├── tests/
│   └── test_service.py
├── Dockerfile
└── requirements.txt
```
