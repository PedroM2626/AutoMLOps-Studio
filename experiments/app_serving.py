
import uvicorn
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import dagshub
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
dagshub.init(repo_owner='PedroM2626', repo_name='experiments', mlflow=True)

app = FastAPI(title="MLOps Enterprise API - rf_hpo_optimized")

# Carregar modelo do MLflow
model_uri = f"models:/rf_hpo_optimized/1"  # Usando versão 1 explicitamente para o teste
model = mlflow.sklearn.load_model(model_uri)

class PredictionInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.text])[0]
    return {"prediction": str(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
