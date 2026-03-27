from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Any
import pandas as pd
import os
from dotenv import load_dotenv
from automl_engine import load_pipeline
from src.tracking.telemetry import TelemetryStore

load_dotenv()

app = FastAPI(title="AutoML Model Serving API")

API_SECRET_KEY = os.getenv("API_SECRET_KEY")
telemetry_store = TelemetryStore()

def verify_api_key(x_api_key: str = Header(...)):
    if not API_SECRET_KEY:
        raise HTTPException(status_code=503, detail="API is not ready: missing API_SECRET_KEY")
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Global storage for the loaded pipeline
model_assets = {"processor": None, "model": None, "model_name": None}

def load_latest_model():
    model_dir = "models"
    if os.path.exists(model_dir):
        files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if files:
            # Load the most recent one
            latest_file = sorted(files)[-1]
            path = os.path.join(model_dir, latest_file)
            processor, model = load_pipeline(path)
            model_assets["processor"] = processor
            model_assets["model"] = model
            model_assets["model_name"] = latest_file
            return True
    return False

@app.on_event("startup")
async def startup_event():
    if not API_SECRET_KEY:
        raise RuntimeError("Missing required environment variable API_SECRET_KEY")
    load_latest_model()

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model_assets["model"] is not None}


@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    ready = bool(API_SECRET_KEY) and model_assets["model"] is not None
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "has_api_key": bool(API_SECRET_KEY),
                "model_loaded": model_assets["model"] is not None,
            },
        )
    return {"status": "ready", "model": model_assets.get("model_name")}

class PredictionRequest(BaseModel):
    data: list[dict[str, Any]]

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(request: PredictionRequest):
    if model_assets["model"] is None:
        if not load_latest_model():
            raise HTTPException(status_code=400, detail="No model loaded. Train a model first.")
    
    try:
        df = pd.DataFrame(request.data)
        X_proc = model_assets["processor"].transform(df)
        predictions = model_assets["model"].predict(X_proc)
        
        # If classifier and label encoder exists, inverse transform
        if hasattr(model_assets["processor"], "label_encoder") and model_assets["processor"].label_encoder:
            pred_classes = model_assets["processor"].label_encoder.inverse_transform(predictions)
            result = pred_classes.tolist()
        else:
            result = predictions.tolist()
            
        # Telemetry logging in SQLite for safe concurrent writes.
        try:
            telemetry_store.log_inference(
                payload_rows=request.data,
                predictions=result,
                model_version=model_assets.get("model_name") or "unknown",
            )
        except Exception as tel_err:
            import logging
            logging.error(f"Failed to log telemetry: {tel_err}")
            
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
