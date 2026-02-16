import pandas as pd
import numpy as np
import os
import sys
from automl_engine import AutoMLDataProcessor, AutoMLTrainer

# ==========================================
# 🚀 AutoMLOps Studio Interface Simulation
# ==========================================

def simulate_automl_run():
    print("🚀 Initializing AutoML Interface Simulation...")
    
    # --- Configuration (User Inputs) ---
    CONFIG = {
        'task_type': 'classification',
        'preset': 'best_quality',
        'seed': 42,
        'target_col': 'sentiment',
        'nlp_col': 'text',
        'nlp_config': {
            'max_features': 50000,
            'ngram_range': (1, 2),
            'vectorizer': 'tfidf',
            'cleaning_mode': 'standard', # Default
            'stop_words': True,
            'lemmatization': False
        },
        'validation_strategy': 'holdout',
        'validation_params': {'test_size': 0.2} # Standard internal split for optimization
    }
    
    print(f"📋 Configuration Loaded: {CONFIG}")
    
    # --- 1. Data Loading ---
    print("\n📂 Loading Datasets...")
    
    # Paths to search for (Prioritize user-specified names, fallback to known location)
    train_paths = [
        'sentiment_train.csv',
        'materialApoio/logistic-senti-pred/data/processed/processed_train.csv'
    ]
    val_paths = [
        'sentiment_validation.csv',
        'materialApoio/logistic-senti-pred/data/processed/processed_validation.csv'
    ]
    
    def load_data(paths, name):
        for p in paths:
            if os.path.exists(p):
                print(f"   ✅ Found {name} at: {p}")
                return pd.read_csv(p)
        print(f"   ❌ Could not find {name} in expected paths: {paths}")
        return None

    train_df = load_data(train_paths, "Train Dataset")
    val_df = load_data(val_paths, "Validation Dataset")
    
    if train_df is None or val_df is None:
        print("🚨 Error: Missing datasets. Please ensure 'sentiment_train.csv' and 'sentiment_validation.csv' exist.")
        sys.exit(1)

    # Ensure columns exist
    required_cols = [CONFIG['target_col'], CONFIG['nlp_col']]
    for col in required_cols:
        if col not in train_df.columns:
            print(f"🚨 Error: Column '{col}' not found in Train Data. Available: {train_df.columns.tolist()}")
            sys.exit(1)
        if col not in val_df.columns:
            print(f"🚨 Error: Column '{col}' not found in Validation Data. Available: {val_df.columns.tolist()}")
            sys.exit(1)
            
    print(f"   Train Shape: {train_df.shape}")
    print(f"   Validation Shape: {val_df.shape}")

    # --- 2. Preprocessing (NLP) ---
    print("\n⚙️ Running AutoML Data Processor...")
    processor = AutoMLDataProcessor(
        target_column=CONFIG['target_col'],
        task_type=CONFIG['task_type'],
        nlp_config=CONFIG['nlp_config']
    )
    
    print("   Processing Train Data (Fit + Transform)...")
    X_train_proc, y_train_proc = processor.fit_transform(train_df, nlp_cols=[CONFIG['nlp_col']])
    
    print("   Processing Validation Data (Transform)...")
    X_val_proc, y_val_proc = processor.transform(val_df)
    
    print(f"   Processed Train Shape: {X_train_proc.shape}")
    print(f"   Processed Validation Shape: {X_val_proc.shape}")

    # --- 3. Training ---
    print(f"\n🤖 Starting AutoML Training (Preset: {CONFIG['preset']})...")
    print("   Note: This may take a while depending on the preset configuration.")
    
    trainer = AutoMLTrainer(
        task_type=CONFIG['task_type'], 
        preset=CONFIG['preset']
    )
    
    # Using 'holdout' strategy for the internal search (split train_df)
    # The external val_df is reserved for final evaluation, simulating 'Test' set behavior
    best_model = trainer.train(
        X_train_proc, 
        y_train_proc,
        validation_strategy=CONFIG['validation_strategy'],
        validation_params=CONFIG['validation_params'],
        random_state=CONFIG['seed'],
        experiment_name="Simulation_SentiPred_GodMode"
    )
    
    print("\n✅ Training Completed!")
    print(f"🏆 Best Model: {trainer.best_params.get('model_name')}")
    print(f"⚙️ Best Hyperparameters: {trainer.best_params}")
    print(f"🏆 Best Score (Internal CV/Val): {trainer.best_score:.4f}")

    # --- 4. Final Evaluation ---
    print("\n🧪 Evaluating on Holdout Validation Set (100% Test)...")
    metrics, y_pred = trainer.evaluate(X_val_proc, y_val_proc)
    
    print("\n📊 Final Results on Validation Set:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"   - {k.upper()}: {v:.4f}")
            
    print("\n🚀 Simulation Finished Successfully.")

if __name__ == "__main__":
    simulate_automl_run()
