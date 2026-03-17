import os
import sys
import time
import pandas as pd
import json
from src.tracking.manager import TrainingJobManager

def get_job_summary(job):
    model_summaries_safe = {}
    for m_name, m_info in (job.model_summaries or {}).items():
        try:
            safe_info = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in m_info.items() if k != "confusion_matrix"}
            if "metrics" in safe_info and isinstance(m_info["metrics"], dict):
                safe_info["metrics"] = {metric_k: (float(metric_v) if isinstance(metric_v, (int, float)) else str(metric_v)) for metric_k, metric_v in m_info["metrics"].items()}
            if "params" in safe_info and isinstance(m_info["params"], dict):
                safe_info["params"] = {param_k: str(param_v) for param_k, param_v in m_info["params"].items()}
            model_summaries_safe[m_name] = safe_info
        except Exception:
            model_summaries_safe[m_name] = {"score": str(m_info.get("score", "-"))}
            
    return json.dumps(model_summaries_safe, default=str)

def main():
    manager = TrainingJobManager()
    
    # Create dummy data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 10,
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10
    })
    
    config = {
        'task': 'classification',
        'target': 'target',
        'train_df': df,
        'preset': 'test',
        'n_trials': 2,
        'timeout': 10,
        'time_budget': 10,
        'experiment_name': 'Debug_Experiment',
        'validation_strategy': 'cv',
        'validation_params': {'folds': 2},
        'optimization_mode': 'random'
    }
    
    print("Submitting job...")
    job_id = manager.submit_job(config)
    
    while manager.has_running_jobs():
        manager.poll_updates()
        time.sleep(1)
        
    job = manager.get_job(job_id)
    print("--- Job Finished ---")
    
    # Run UI processing
    summary_json = get_job_summary(job)
    print(f"Summary JSON:\n{summary_json}")
    
    try:
        parsed = json.loads(summary_json)
        print("\nSuccessfully parsed JSON!")
        print(f"Parsed Keys: {list(parsed.keys())}")
    except Exception as e:
        print(f"\nJSON parsing failed: {e}")

if __name__ == "__main__":
    main()
