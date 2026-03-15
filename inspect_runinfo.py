import mlflow
from mlflow.entities import RunInfo
import inspect

print(f"MLflow Version: {mlflow.__version__}")
sig = inspect.signature(RunInfo.__init__)
print(f"RunInfo.__init__ signature: {sig}")

for name, param in sig.parameters.items():
    print(f"  Parameter: {name}, Kind: {param.kind}, Default: {param.default}")

# Try to instantiate RunInfo with only run_id
try:
    print("\nAttempting to instantiate RunInfo(run_id='test')...")
    # Note: official MLflow 2.x uses run_id, 1.x used run_uuid
    ri = RunInfo(run_id="test_id", experiment_id="0", user_id="user", status="FINISHED", start_time=0, end_time=1, artifact_uri="uri", lifecycle_stage="active")
    print("Success!")
except Exception as e:
    print(f"FAILED: {e}")
