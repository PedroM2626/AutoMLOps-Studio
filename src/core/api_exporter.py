import os
import shutil
import tempfile
import zipfile
import mlflow
from mlflow.tracking import MlflowClient

def export_model_api(model_name: str, version: str) -> str:
    """
    Exports a registered model as a self-contained FastAPI + Docker zip bundle.
    Returns the path to the generated zip file.
    """
    client = MlflowClient()
    
    # 1. Get model version details to find the artifact URI
    try:
        model_version_details = client.get_model_version(name=model_name, version=version)
        run_id = model_version_details.run_id
    except Exception as e:
        raise ValueError(f"Failed to fetch model details from MLflow Registry: {e}")
        
    # 2. Download the model artifacts
    temp_dir = tempfile.mkdtemp(prefix="automl_api_bundle_")
    
    try:
        # MLflow's download_artifacts fetches the whole folder
        model_path = client.download_artifacts(run_id, "model", dst_path=temp_dir)
        
        # 3. Create app.py (FastAPI Server)
        app_code = f"""import os
import pandas as pd
from fastapi import FastAPI, Request
import mlflow.pyfunc
import uvicorn

app = FastAPI(title="AutoMLOps Model API - {model_name} (v{version})", version="{version}")

# The model folder will be mounted at the same level as app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

print(f"Loading model from {{MODEL_PATH}}...")
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {{e}}")
    model = None

@app.get("/health")
def health_check():
    return {{"status": "Healthy", "model_loaded": model is not None}}

@app.post("/predict")
async def predict(request: Request):
    if model is None:
        return {{"error": "Model not loaded properly."}}
        
    data = await request.json()
    try:
        # Attempt to handle both single dictionary or list of dictionaries
        if isinstance(data, dict):
            # Might be formatted as {{"feature1": val, ...}} or {{"data": [...]}}
            if "data" in data and isinstance(data["data"], list):
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return {{"error": "Invalid payload format. Send JSON object or array."}}
            
        predictions = model.predict(df)
        
        # Format response natively
        if hasattr(predictions, "tolist"):
            res = predictions.tolist()
        else:
            res = list(predictions)
            
        return {{"predictions": res}}
        
    except Exception as e:
        import traceback
        return {{"error": str(e), "trace": traceback.format_exc()}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open(os.path.join(temp_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_code)
            
        # 4. Read the MLflow requirements and merge with API requirements
        req_path = os.path.join(model_path, "requirements.txt")
        api_reqs = ["fastapi", "uvicorn", "pydantic", "pandas"]
        if os.path.exists(req_path):
            with open(req_path, "r", encoding="utf-8") as f:
                model_reqs = [line.strip() for line in f.readlines() if line.strip()]
            
            # Merge and deduplicate
            final_reqs = set(model_reqs)
            for ar in api_reqs:
                if not any(r.startswith(ar) for r in final_reqs):
                    final_reqs.add(ar)
        else:
            # Fallback
            final_reqs = set(api_reqs + ["mlflow", "scikit-learn"])
            
        with open(os.path.join(temp_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(final_reqs)))
            
        # 5. Create Dockerfile
        dockerfile_code = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required by some ML libraries (like lightgbm)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        with open(os.path.join(temp_dir, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(dockerfile_code)
            
        # 6. Create ZIP bundle
        zip_filename = f"{model_name}_v{version}_api.zip"
        zip_filepath = os.path.join(tempfile.gettempdir(), zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
                    
        return zip_filepath
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
