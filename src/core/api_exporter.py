import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

from mlflow.tracking import MlflowClient

APP_CODE = """import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
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
        raise HTTPException(status_code=503, detail="Model not loaded properly.")

    try:
        data = await request.json()
    except Exception as invalid_body:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {{invalid_body}}")

    # Accept a single record, an array of records, or {{"data": [...]}}
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            records = data["data"]
        else:
            records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise HTTPException(status_code=400, detail="Send a JSON object, an array, or a {{'data': [...]}} body.")

    if not records:
        raise HTTPException(status_code=400, detail="Empty payload: no rows to score.")

    try:
        df = pd.DataFrame(records)
        predictions = model.predict(df)
    except (ValueError, KeyError, TypeError) as invalid_input:
        # Missing or malformed columns are the caller's problem, not a server fault.
        raise HTTPException(status_code=400, detail=str(invalid_input))
    except Exception as failure:
        # mlflow reports a schema mismatch as MlflowException("Failed to enforce schema..."),
        # which is a bad request, not a server fault.
        detail = str(failure)
        if isinstance(failure, MlflowException) and "enforce schema" in detail:
            raise HTTPException(status_code=400, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    else:
        predictions = list(predictions)

    return {{"predictions": predictions}}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

DOCKERFILE_CODE = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required by some ML libraries (like lightgbm)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

API_REQUIREMENTS = ["fastapi", "uvicorn", "pydantic", "pandas"]


def _validate_ref(model_name: str, version: str) -> None:
    import re
    if not re.match(r'^[\w-]+$', str(model_name)):
        raise ValueError("Invalid model_name. Only alphanumeric characters, dashes, and underscores are allowed.")
    if not re.match(r'^[\w.-]+$', str(version)):
        raise ValueError("Invalid version. Only alphanumeric characters, dots, dashes, and underscores are allowed.")


def build_bundle_dir(model_name: str, version: str, dest_dir: str) -> str:
    """Materialise the self-contained FastAPI + Docker service for a registered model.

    Shared by the downloadable zip and the local service launcher so the two cannot
    drift apart.
    """
    _validate_ref(model_name, version)
    client = MlflowClient()

    try:
        run_id = client.get_model_version(name=model_name, version=version).run_id
    except Exception as e:
        raise ValueError(f"Failed to fetch model details from MLflow Registry: {e}")

    # MLflow's download_artifacts fetches the whole folder
    model_path = client.download_artifacts(run_id, "model", dst_path=dest_dir)

    with open(os.path.join(dest_dir, "app.py"), "w", encoding="utf-8") as handle:
        handle.write(APP_CODE.format(model_name=model_name, version=version))

    req_path = os.path.join(model_path, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as handle:
            final_reqs = {line.strip() for line in handle if line.strip()}
        for api_req in API_REQUIREMENTS:
            if not any(req.startswith(api_req) for req in final_reqs):
                final_reqs.add(api_req)
    else:
        final_reqs = set(API_REQUIREMENTS + ["mlflow", "scikit-learn"])

    with open(os.path.join(dest_dir, "requirements.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sorted(final_reqs)))

    with open(os.path.join(dest_dir, "Dockerfile"), "w", encoding="utf-8") as handle:
        handle.write(DOCKERFILE_CODE)

    return dest_dir


def export_model_api(model_name: str, version: str) -> str:
    """
    Exports a registered model as a self-contained FastAPI + Docker zip bundle.
    Returns the path to the generated zip file.
    """
    temp_dir = tempfile.mkdtemp(prefix="automl_api_bundle_")
    try:
        build_bundle_dir(model_name, version, temp_dir)
        zip_filepath = os.path.join(tempfile.gettempdir(), f"{model_name}_v{version}_api.zip")

        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as archive:
            for root, _dirs, files in os.walk(temp_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    archive.write(file_path, os.path.relpath(file_path, temp_dir))
        return zip_filepath
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_local_service(model_name: str, version: str, timeout: float = 150.0) -> dict:
    """Serve a registered model for real: build the bundle, run uvicorn, poll /health.

    Returns a handle with pid, url and the service's own health payload. Raises
    RuntimeError carrying the service output when it never becomes healthy, so the
    caller reports what actually happened instead of a made-up status.
    """
    import urllib.request

    service_dir = tempfile.mkdtemp(prefix="automl_service_")
    build_bundle_dir(model_name, version, service_dir)
    port = _free_port()
    log_path = os.path.join(service_dir, "service.log")
    log_handle = open(log_path, "w", encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=service_dir,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    last_error = "service never answered /health"
    while time.time() < deadline:
        if process.poll() is not None:
            last_error = f"service exited with code {process.returncode}"
            break
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as response:
                health = json.loads(response.read().decode("utf-8"))
            log_handle.close()
            return {
                "pid": process.pid,
                "process": process,
                "url": url,
                "health": health,
                "dir": service_dir,
                "log_path": log_path,
                "model": model_name,
                "version": str(version),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0)

    log_text = ""
    try:
        log_handle.close()
        with open(log_path, "r", encoding="utf-8") as handle:
            log_text = handle.read()[-1500:]
    except OSError:
        pass
    stop_local_service({"process": process, "dir": service_dir})
    raise RuntimeError(f"Local service did not become healthy ({last_error}). Service output:\n{log_text}")


def stop_local_service(handle: dict) -> None:
    """Terminate the uvicorn process started by start_local_service and remove its dir."""
    if not isinstance(handle, dict):
        return
    process = handle.get("process")
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    service_dir = handle.get("dir")
    if service_dir and os.path.isdir(service_dir):
        shutil.rmtree(service_dir, ignore_errors=True)
