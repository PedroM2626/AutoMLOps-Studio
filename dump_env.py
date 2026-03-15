import sys
import os
import mlflow
import inspect

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"sys.path: {sys.path}")

print(f"\nMLflow version: {mlflow.__version__}")
print(f"MLflow file: {mlflow.__file__}")

try:
    from mlflow.entities import RunInfo
    print(f"RunInfo file: {inspect.getfile(RunInfo)}")
    print(f"RunInfo init signature: {inspect.signature(RunInfo.__init__)}")
except Exception as e:
    print(f"Error inspecting RunInfo: {e}")

# Check for other mlflow packages in site-packages
import site
for s in site.getsitepackages():
    if os.path.exists(s):
        print(f"\nChecking site-package: {s}")
        for item in os.listdir(s):
            if "mlflow" in item.lower():
                print(f"  Found: {item}")
