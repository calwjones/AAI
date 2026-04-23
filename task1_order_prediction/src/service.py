from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from prediction import OrderPredictor

predictor = OrderPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="BRFN Order Prediction Service", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    customer_id: int
    top_n: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "order-prediction",
        "model_loaded": predictor.is_loaded(),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    recommendations = predictor.recommend(request.customer_id, top_n=request.top_n)
    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail=f"No order history found for customer {request.customer_id}.",
        )

    return {
        "customer_id": request.customer_id,
        "recommendations": recommendations,
    }


@app.get("/customers")
def customers():
    return {"customer_ids": predictor.known_customers()}


@app.get("/metadata")
def metadata():
    return predictor.metadata

@app.get("/forecast")
def forecast(days: int = 7):
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor.forecast(days=days)

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8002, reload=True)
