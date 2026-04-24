# Tasks 2, 3, 4 — Quality Grading, Model Management, and XAI

A single FastAPI service covering three of the case study tasks:

- **Task 2** — Product quality grading (fresh/rotten CNN + Color/Size/Ripeness scoring layer + Grade A/B/C)
- **Task 3** — AI engineer model upload and interaction logging (hot-swap without redeploy, every prediction logged to DESD)
- **Task 4** — Explainable AI (Grad-CAM heatmaps on the quality model). The admin-facing side of Task 4 lives in [`../dashboard/`](../dashboard/README.md).

**Port:** 8001

## Task 2 — Quality Grading

### Pipeline

1. Image uploaded as multipart form (image + `product_id` + optional `user_id`)
2. Preprocessed: RGB, resized to 224×224, normalised to `[0, 1]`
3. Active classifier predicts Fresh vs Rotten
4. Three sub-scores computed from image properties:
   - **Color** — HSV saturation (weighted 0.7) + brightness penalty (weighted 0.3), blended 50/50 with classifier confidence
   - **Size** — Otsu foreground ratio with three regimes: `<0.2` penalised (too small), `>0.85` penalised (cropped too tight), middle range scaled 70–100
   - **Ripeness** — RGB warmth `(R + 0.5·G) / B`, blended 60/40 with classifier confidence
5. Letter grade from case-study thresholds:
   - **Grade A:** Color ≥75, Size ≥80, Ripeness ≥70
   - **Grade B:** Color ≥65, Size ≥70, Ripeness ≥60
   - **Grade C:** below B

## Task 3 — Model Upload & Interaction Logging

### Upload and hot-swap

`ModelManager` writes each upload to `models/` with a timestamped filename, appends metadata to `versions.json`, and loads the new model into memory in-process. On startup the latest version by `uploaded_at` is activated. New uploads are active immediately — no container restart.

Supported formats:

| Extension | Framework | Prediction call |
|---|---|---|
| `.keras`, `.h5` | TensorFlow/Keras | `model.predict(batch)` on `(1, 224, 224, 3)` tensor |
| `.pkl` | scikit-learn | `model.predict_proba(flat)` on flattened vector |

The two interfaces can't be reconciled by a single abstraction (different input shapes, different method names), so format-specific branching lives inside `QualityGrader.grade()`.

### DESD audit logging

`InteractionLogger` authenticates against DESD with a session login (GET `/accounts/login/` → CSRF token → POST with `csrfmiddlewaretoken` and `Referer` header) and attaches `X-CSRFToken` on every log POST. Payload sent to `/api/ai-logs/`:

```json
{
  "service_type": "quality",
  "user": user_id,
  "input_data": {"product_id": ..., "filename": "..."},
  "prediction": {...},
  "model_version": "...",
  "confidence_score": 0.94,
  "user_override": false
}
```

This gives the wider team a single audit trail across both AI services rather than each keeping its own logs.

## Task 4 — Explainable AI

`/explain` runs Grad-CAM on the quality classifier's last convolutional layer (`out_relu` of MobileNetV2). The gradient of the predicted class with respect to the feature maps weights a per-pixel sum, producing a heatmap of which image regions drove the decision. The heatmap is overlaid on the original image with a Jet colourmap, encoded as base64 PNG, and returned alongside a plain-English region description (e.g. *"top-left region"*).

The dashboard at [`../dashboard/`](../dashboard/README.md) consumes `/explain` and renders the heatmap inline.

## API

| Method | Path | Task | Description |
|---|---|---|---|
| `GET` | `/health` | — | Status + active model version |
| `POST` | `/grade` | 2 | Image → fresh/rotten + grade + sub-scores |
| `POST` | `/explain` | 4 | Image → Grad-CAM heatmap + assessment |
| `POST` | `/upload-model` | 3 | Upload `.keras`, `.h5`, or `.pkl` with version metadata |
| `GET` | `/models` | 3 | List all uploaded versions |
| `GET` | `/interactions` | 3 | Fetch DESD audit logs (filterable by service, date, override) |

Example upload:

```bash
curl -X POST http://localhost:8001/upload-model \
  -F "model_file=@best_model_phase2.keras" \
  -F "version=v2_phase2" \
  -F "accuracy=0.991" \
  -F "notes=MobileNetV2 fine-tuned"
```

Example grade:

```bash
curl -X POST http://localhost:8001/grade \
  -F "image=@apple.jpg" \
  -F "product_id=1" \
  -F "user_id=42"
```

Interactive docs at http://localhost:8001/docs.

## Running

Via docker-compose (from repo root):

```bash
docker compose up ai-quality-service --build
```

Standalone (Python 3.11, TensorFlow installed):

```bash
pip install -r requirements.txt
cd src && uvicorn service:app --port 8001 --reload
```

## Registering a model without HTTP

`scripts/register_model.py` injects a model into `versions.json` directly on disk — useful during development when you don't want to start the service just to upload.

```bash
python scripts/register_model.py models/best_model_phase2.keras v1_phase2 \
  --accuracy 0.991 \
  --notes "MobileNetV2"
```

## Tests

```bash
pytest tests/
```

`conftest.py` monkey-patches `ModelManager`'s storage paths to a tempdir at import time, so test runs can't pollute `versions.json` or overwrite the live model. Ten tests cover health, upload validation (good `.pkl`, rejected bad extensions), model listing, grade response structure, A/B/C threshold logic, image validation, and `/interactions`. DESD is patched at the test boundary.

## Layout

```
task2_3_4_quality_xai/
├── src/
│   ├── service.py            # FastAPI app
│   ├── quality_grader.py     # Scoring + grade mapping (Task 2)
│   ├── explainer.py          # Grad-CAM (Task 4)
│   ├── model_upload.py       # ModelManager (Task 3)
│   └── interaction_logger.py # DESD client (Task 3)
├── models/
│   ├── best_model_phase2.keras
│   └── versions.json
├── scripts/
│   └── register_model.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
├── tests/
│   ├── conftest.py
│   └── test_service.py
├── Dockerfile
└── requirements.txt
```
