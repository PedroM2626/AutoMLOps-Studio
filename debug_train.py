
import pandas as pd
import numpy as np
from automl_engine import AutoMLTrainer, AutoMLDataProcessor
import logging

logging.basicConfig(level=logging.INFO)

def test_train():
    # Create a simple dataset
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    df = pd.DataFrame(X, columns=[f'col_{i}' for i in range(10)])
    df['target'] = y
    
    processor = AutoMLDataProcessor(target_column='target', task_type='classification')
    X_proc, y_proc = processor.fit_transform(df)
    
    trainer = AutoMLTrainer(task_type='classification')
    
    def callback(trial, score, name, duration, metrics):
        print(f"Finished {name} with score {score:.4f} in {duration:.2f}s")

    print("Starting training...")
    # Test only svm and linear_svc to see if it hangs between them
    best_model = trainer.train(
        X_proc, 
        y_proc, 
        n_trials=2, 
        selected_models=['svm', 'linear_svc', 'knn'], 
        callback=callback
    )
    print("Training finished successfully!")

if __name__ == "__main__":
    test_train()
