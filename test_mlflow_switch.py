
import mlflow
import os
import shutil
import time

def test_uri_switch():
    print("--- Starting MLflow URI Switch Test ---")

    # 1. Start with Local SQLite
    local_db = "mlflow_test_local.db"
    local_uri = f"sqlite:///{local_db}"
    
    # Clean up previous run
    if os.path.exists(local_db):
        os.remove(local_db)
        
    print(f"\n[1] Setting Local URI: {local_uri}")
    os.environ["MLFLOW_TRACKING_URI"] = local_uri
    mlflow.set_tracking_uri(local_uri)
    
    print(f"Current URI: {mlflow.get_tracking_uri()}")
    assert mlflow.get_tracking_uri() == local_uri
    
    # Create a local experiment
    exp_name_local = "Test_Local_Switch"
    try:
        mlflow.create_experiment(exp_name_local)
    except:
        pass
    mlflow.set_experiment(exp_name_local)
    
    with mlflow.start_run() as run:
        mlflow.log_param("location", "local")
        print(f"Logged run {run.info.run_id} to local.")

    # 2. Switch to 'Remote' (Simulated with another SQLite DB to avoid needing real credentials)
    # In a real scenario, this would be the DagsHub URI
    remote_db = "mlflow_test_remote.db"
    remote_uri = f"sqlite:///{remote_db}"
    
    if os.path.exists(remote_db):
        os.remove(remote_db)

    print(f"\n[2] Switching to 'Remote' URI: {remote_uri}")
    os.environ["MLFLOW_TRACKING_URI"] = remote_uri
    mlflow.set_tracking_uri(remote_uri)
    
    print(f"Current URI: {mlflow.get_tracking_uri()}")
    assert mlflow.get_tracking_uri() == remote_uri
    
    # Create a remote experiment
    exp_name_remote = "Test_Remote_Switch"
    try:
        mlflow.create_experiment(exp_name_remote)
    except:
        pass
    mlflow.set_experiment(exp_name_remote)
    
    with mlflow.start_run() as run:
        mlflow.log_param("location", "remote")
        print(f"Logged run {run.info.run_id} to remote.")

    # 3. Verify Separation
    print("\n[3] Verifying Data Separation...")
    
    # Check Local
    mlflow.set_tracking_uri(local_uri)
    client_local = mlflow.MlflowClient()
    runs_local = client_local.search_runs(
        experiment_ids=[client_local.get_experiment_by_name(exp_name_local).experiment_id]
    )
    print(f"Local Runs: {len(runs_local)}")
    assert len(runs_local) == 1
    assert runs_local[0].data.params['location'] == 'local'

    # Check Remote
    mlflow.set_tracking_uri(remote_uri)
    client_remote = mlflow.MlflowClient()
    runs_remote = client_remote.search_runs(
        experiment_ids=[client_remote.get_experiment_by_name(exp_name_remote).experiment_id]
    )
    print(f"Remote Runs: {len(runs_remote)}")
    assert len(runs_remote) == 1
    assert runs_remote[0].data.params['location'] == 'remote'
    
    print("\n✅ URI Switching Test Passed Successfully!")
    
    # Cleanup
    if os.path.exists(local_db):
        os.remove(local_db)
    if os.path.exists(remote_db):
        os.remove(remote_db)

if __name__ == "__main__":
    test_uri_switch()
