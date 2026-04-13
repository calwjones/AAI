from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="BRFN Order Prediction Service", version="1.0.0")


class PredictRequest(BaseModel):
    customer_id: int
    top_n: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "service": "order-prediction"}


@app.post("/predict")
def predict(request: PredictRequest):
    return {
        "customer_id": request.customer_id,
        "recommendations": [],
    }


if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8002, reload=True)
