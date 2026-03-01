import pandas as pd
from sklearn.datasets import make_classification
from automl_engine import AutoMLDataProcessor, AutoMLTrainer
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

try:
    print("Generating data...")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'f_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    
    print("Processing...")
    processor = AutoMLDataProcessor()
    X_proc, y_proc = processor.fit_transform(X, y)
    
    print("Training...")
    trainer = AutoMLTrainer(task_type='classification', preset='fast')
    trainer.train(X_proc, y_proc, feature_names=processor.get_feature_names())
    
    print('\n--- RESULTS ---')
    best_m = trainer.best_params.get('model_name')
    print('Best Model:', best_m)
    
    if hasattr(trainer, 'model_summaries'):
        metrics = trainer.model_summaries[best_m].get('metrics', {})
        print('Model Card Gen:', 'model_card' in metrics)
        if 'model_card' in metrics:
            print("\n----- CARD PREVIEW -----")
            print(metrics['model_card'][:300])
except Exception as e:
    import traceback
    traceback.print_exc()
