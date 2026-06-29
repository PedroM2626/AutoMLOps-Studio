import numpy as np
import pandas as pd
from src.core.processor import AutoMLDataProcessor
from src.engines.classical import AutoMLTrainer

def test_temporal_nlp_processor():
    print("Testing temporal and NLP data characteristics processing...")
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'text': ["Hello world", "This is text", "Another sentence", "Machine learning", "AutoML ops", 
                 "Hello world", "This is text", "Another sentence", "Machine learning", "AutoML ops"],
        'numeric': [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5, 3.0, 2.5],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # 1. Test Processor with Temporal and NLP enabled
    processor = AutoMLDataProcessor(
        target_column='target',
        task_type='classification',
        date_col='date',
        data_type='sequential',
        nlp_config={'vectorizer': 'passthrough'}, # passthrough vectorizer
        scaler_type='standard'
    )
    
    X_proc, y_proc = processor.fit_transform(df, nlp_cols=['text'])
    print(f"Processed shape: {X_proc.shape}")
    assert X_proc.shape[0] > 0
    assert y_proc is not None
    print("Processor test passed!")

def test_forecast_trainer():
    print("Testing Forecast task type...")
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
        'target': np.sin(np.linspace(0, 10, 15))
    })
    
    processor = AutoMLDataProcessor(
        target_column='target',
        task_type='forecast',
        date_col='date',
        data_type='sequential'
    )
    X_proc, y_proc = processor.fit_transform(df)
    
    trainer = AutoMLTrainer(task_type='forecast', preset='test', use_ensemble=False)
    # Fit
    trainer.train(X_proc, y_proc, n_trials=1, validation_strategy='holdout', validation_params={'test_size': 0.2})
    print("Forecast trainer test passed!")

def test_multitask_trainer():
    print("Testing Multi-Task Classification...")
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 2, size=(20, 2))
    
    trainer = AutoMLTrainer(task_type='multi_task', preset='test', use_ensemble=False)
    trainer.train(X, y, n_trials=1, validation_strategy='holdout')
    print("Multi-task trainer test passed!")

def test_semi_supervised_trainer():
    print("Testing Semi-Supervised Classification...")
    df = pd.DataFrame({
        'feat1': np.random.randn(30),
        'feat2': np.random.randn(30),
        'target': [0, 1, 0, 1, 0, -1, -1, -1, 1, 0] * 3  # -1 represents unlabeled
    })
    
    processor = AutoMLDataProcessor(
        target_column='target',
        task_type='classification',
        semi_supervised=True
    )
    X_proc, y_proc = processor.fit_transform(df)
    assert -1 in y_proc
    
    trainer = AutoMLTrainer(task_type='classification', preset='test', use_ensemble=False, semi_supervised=True)
    trainer.train(X_proc, y_proc, n_trials=1, validation_strategy='holdout')
    print("Semi-supervised trainer test passed!")

if __name__ == "__main__":
    test_temporal_nlp_processor()
    test_forecast_trainer()
    test_multitask_trainer()
    test_semi_supervised_trainer()
    print("All AutoMLOps-Studio tests passed!")
