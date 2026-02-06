from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from automl_engine import load_pipeline

load_dotenv()

app = FastAPI(title="AutoML Model Serving API")

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "supersecretkey123")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Global storage for the loaded pipeline
model_assets = {"processor": None, "model": None}

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
            return True
    return False

@app.on_event("startup")
async def startup_event():
    load_latest_model()

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model_assets["model"] is not None}

class PredictionRequest(BaseModel):
    data: list # List of dicts for rows

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
            predictions = model_assets["processor"].label_encoder.inverse_transform(predictions)
            
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
