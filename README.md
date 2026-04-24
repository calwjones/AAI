# Advanced AI — BRFN

**Module:** UFCFUR-15-3 (Advanced AI)
**Client:** Bristol Regional Food Network (BRFN)
**Submission:** 23rd April 2026

AI microservices for the DESD Marketplace — a platform connecting local producers to customers. Two FastAPI services and a React dashboard, containerised with Docker and integrated with the DESD platform for authentication and audit logging.

## Case study tasks

The case study defines four tasks. They map to two service folders:

| Task | What | Folder |
|---|---|---|
| **Task 1** — Intelligent Order Prediction | Analyses customer purchase history and suggests re-orders | [`task1_order_prediction/`](task1_order_prediction/README.md) |
| **Task 2** — Product Quality Grading | Classifies fruit/veg images as fresh or rotten, returns Grade A/B/C with Color/Size/Ripeness breakdown | [`task2_3_4_quality_xai/`](task2_3_4_quality_xai/README.md) |
| **Task 3** — Model Upload & Interaction Logging | Lets AI engineers upload new models without redeploy; logs every prediction to DESD | [`task2_3_4_quality_xai/`](task2_3_4_quality_xai/README.md) |
| **Task 4** — Explainable AI & Admin Dashboard | Grad-CAM explanations + dashboard showing predictions, confidence, and accuracy over time | [`task2_3_4_quality_xai/`](task2_3_4_quality_xai/README.md) + [`dashboard/`](dashboard/README.md) |

## Architecture

```
Dashboard (React/Vite, 5173) ──► Quality + XAI Service ──┐
                                 (FastAPI, 8001)          │
                                                          ├─► DESD API (8089)
                                 Order Prediction         │   /api/ai-logs/
                                 (FastAPI, 8002) ─────────┘
```

Every prediction from either service is logged to DESD so the team has a single audit trail across both services rather than each one keeping its own.

## Running it

```bash
# one-time: create the shared network DESD lives on
docker network create brfn-shared

# set env
cp .env.example .env

# build and run
docker compose up --build
```

- Dashboard — http://localhost:5173
- Quality API — http://localhost:8001 (interactive docs at `/docs`)
- Order Prediction API — http://localhost:8002 (interactive docs at `/docs`)

## Environment

`.env` needs:

```
DESD_API_URL=http://host.docker.internal:8089/api
DESD_SERVICE_USERNAME=admin
DESD_SERVICE_PASSWORD=demo1234
```

`host.docker.internal` lets containers reach the DESD instance running on the host (configured via `extra_hosts` in `docker-compose.yml`).

## Tech

Python 3.11, FastAPI, scikit-learn, TensorFlow/Keras, OpenCV, React 19, Vite, Docker Compose.

## Team

- **Cal** — Task 3 (model upload, interaction logging, DESD integration), repo scaffolding
- **Tommy** — Task 1 order prediction model
- **Charles** — Task 2 quality classification CNN
- **Arran** — Task 4 XAI dashboard, Docker networking

## Docs

- [`docs/genai_log.md`](docs/genai_log.md) — GenAI usage log (10% of marks)
- Per-service READMEs linked in the task table above
