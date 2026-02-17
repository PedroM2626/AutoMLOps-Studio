import pandas as pd
import numpy as np
import os
import sys
from automl_engine import AutoMLDataProcessor, AutoMLTrainer, TransformersWrapper

# ==========================================
# 🚀 AutoMLOps Studio Interface Simulation
# ==========================================

import sys
print("DEBUG: Script started", flush=True)
# sys.stdout.reconfigure(encoding='utf-8')
sys.stderr = sys.stdout

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
            'ngram_range': (1, 3), # God Mode uses (1, 3)
            'vectorizer': 'tfidf',
            'cleaning_mode': 'god_mode', # Enable God Mode cleaning
            'stop_words': False, # God Mode does NOT remove stop words (crucial for sentiment)
            'lemmatization': False,
            'sublinear_tf': True, # Enable sublinear TF scaling
            'strip_accents': 'unicode' # Handle accents
        },
        'validation_strategy': 'holdout',
        'validation_params': {
            'test_size': 0.2
        }
    }

    # Optimization for performance proof:
    # Reduce trials to 1 because VotingEnsemble has fixed hyperparameters (God Mode)
    # and we want to use more data without waiting forever.
    CONFIG['n_trials_override'] = 1
    
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

    # --- Filter Columns (Avoid Garbage Features) ---
    # Only keep Target, NLP column, and 'entity' if present. Drop IDs and auxiliary text columns.
    keep_cols = [CONFIG['target_col'], CONFIG['nlp_col']]
    if 'entity' in train_df.columns:
        keep_cols.append('entity')
    
    print(f"   Filtering columns to: {keep_cols}")
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]

    # Ensure columns exist
    required_cols = [CONFIG['target_col'], CONFIG['nlp_col']]
    for col in required_cols:
        if col not in train_df.columns:
            print(f"🚨 Error: Column '{col}' not found in Train Data. Available: {train_df.columns.tolist()}")
            sys.exit(1)
        if col not in val_df.columns:
            print(f"🚨 Error: Column '{col}' not found in Validation Data. Available: {val_df.columns.tolist()}")
            sys.exit(1)

    # Drop NaNs in target to ensure alignment
    train_df = train_df.dropna(subset=[CONFIG['target_col']])
    val_df = val_df.dropna(subset=[CONFIG['target_col']])

    # Subsample for simulation speed (Transformers on CPU are slow)
    # INCREASED for better performance verification
    # Using 30000 rows to approximate God Mode performance (full dataset is ~74k)
    # This should yield > 80% accuracy
    if len(train_df) > 200:
        print(f"⚠️ Subsampling Train Data from {len(train_df)} to 200 for better representation.")
        train_df = train_df.sample(n=200, random_state=CONFIG['seed'])
    
    # Drop duplicates in text to avoid leakage/bias (Critical for God Mode performance)
    print(f"   Removing duplicates from Train Data...")
    train_df = train_df.drop_duplicates(subset=[CONFIG['nlp_col']])
            
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
    
    if 'n_trials_override' in CONFIG:
        print(f"   Overriding n_trials to {CONFIG['n_trials_override']}")
        trainer.preset_configs['best_quality']['n_trials'] = CONFIG['n_trials_override']
        
        # Override models to include custom ensembles
        trainer.preset_configs['best_quality']['models'] = ['voting_ensemble', 'custom_voting', 'custom_stacking']
        print(f"   Overriding models to ['voting_ensemble', 'custom_voting', 'custom_stacking'] for feature verification.")
    
    # Define Custom Ensemble Config (Simulating UI selection with STRINGS)
    # from sklearn.linear_model import LogisticRegression, SGDClassifier
    # from sklearn.ensemble import RandomForestClassifier
    
    ensemble_config = {
        'voting_estimators': ['logistic_regression', 'random_forest'],
        'voting_type': 'soft',
        'voting_weights': [2.0, 1.0],
        'stacking_estimators': ['random_forest', 'svm'],
        'stacking_final_estimator': 'logistic_regression' # Testing string resolution
    }

    # Using 'holdout' strategy for the internal search (split train_df)
    # The external val_df is reserved for final evaluation, simulating 'Test' set behavior
    try:
        best_model = trainer.train(
            X_train_proc, 
            y_train_proc,
            validation_strategy=CONFIG['validation_strategy'],
            validation_params=CONFIG['validation_params'],
            random_state=CONFIG['seed'],
            experiment_name="Simulation_SentiPred_GodMode",
            X_raw=train_df[CONFIG['nlp_col']], # Pass raw text for Transformers
            ensemble_config=ensemble_config
        )
    except Exception as e:
        print(f"🚨 Training Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Training Completed!")
    best_model_name = trainer.best_params.get('model_name')
    print(f"🏆 Best Model: {best_model_name}")
    print(f"⚙️ Best Hyperparameters: {trainer.best_params}")
    
    # Retrieve best score safely
    best_score = 0.0
    if hasattr(trainer, 'study') and trainer.study is not None:
         try:
             best_score = trainer.study.best_value
         except:
             pass
    
    if best_score is None or best_score == 0.0:
        # Fallback to model_summaries if study is not available
        if hasattr(trainer, 'model_summaries') and trainer.model_summaries:
             try:
                 best_score = max([m['score'] for m in trainer.model_summaries.values()])
             except:
                 pass

    print(f"🏆 Best Score (Internal Optimization): {best_score:.4f}")
    
    # Print summary of all models
    print("\n📊 Model Comparison (Internal Scores):")
    if hasattr(trainer, 'model_summaries'):
        for m_name, summary in trainer.model_summaries.items():
            score_val = summary.get('score', 0.0)
            print(f"   - {m_name}: {score_val:.4f}")

    # --- 4. Final Evaluation ---
    print("\n🧪 Evaluating on Holdout Validation Set (100% Test)...")
    
    # Determine input for evaluation (Vectorized or Raw)
    X_eval = X_val_proc
    if isinstance(best_model, TransformersWrapper):
         print(f"ℹ️ Best model is a Transformer ({best_model_name}). Using raw text for evaluation.")
         X_eval = val_df[CONFIG['nlp_col']]
    
    # Check if evaluate method exists or use predict directly
    if hasattr(trainer, 'evaluate'):
        try:
            metrics, _ = trainer.evaluate(X_eval, y_val_proc)
            print("\n📊 Final Results on Validation Set:")
            for k, v in metrics.items():
                if k != 'confusion_matrix':
                    print(f"   - {k.upper()}: {v:.4f}")
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            # Fallback to manual prediction
            y_pred = best_model.predict(X_eval)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_val_proc, y_pred)
            print(f"   - ACCURACY: {acc:.4f}")
    else:
        y_pred = best_model.predict(X_eval)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_val_proc, y_pred)
        print(f"   - ACCURACY: {acc:.4f}")
            
    print("\n🚀 Simulation Finished Successfully.")

if __name__ == "__main__":
    simulate_automl_run()
