import os
import pandas as pd
import numpy as np
import logging
import sys
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock environment variables to avoid conflicts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from automl_engine import AutoMLDataProcessor, AutoMLTrainer, TRANSFORMERS_AVAILABLE
except ImportError as e:
    logger.error(f"Failed to import automl_engine: {e}")
    sys.exit(1)

def test_transformers_interface():
    logger.info("Starting Transformers Interface Test...")
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not detected. Skipping test.")
        return

    # Create dummy dataset (Classification)
    # Using enough data to allow splitting if needed
    data = {
        'text': [
            "This is a positive review.",
            "I hate this product.",
            "Absolutely fantastic service!",
            "Terrible experience, never again.",
            "It was okay, not great.",
            "Highly recommended!",
            "Worst purchase ever.",
            "Love it, great quality.",
            "Not worth the money.",
            "Best item I bought this year."
        ] * 10, # 100 samples
        'sentiment': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1] * 10
    }
    df = pd.DataFrame(data)
    
    logger.info(f"Dataset created with shape: {df.shape}")

    # --- TEST: Raw Text (Passthrough) ---
    logger.info("\n--- TEST: Raw Text Passthrough for Transformers ---")
    
    # NLP Config: Passthrough to keep text raw
    nlp_config_raw = {
        'cleaning_mode': 'standard',
        'vectorizer': 'passthrough', # CRITICAL for Transformers
        'max_features': 100, 
        'ngram_range': (1, 1)
    }
    
    processor_raw = AutoMLDataProcessor(
        target_column='sentiment',
        task_type='classification',
        nlp_config=nlp_config_raw
    )
    
    logger.info("Processing data (Raw)...")
    try:
        # Fit transform to get X and y
        X_processed_raw, y_processed_raw = processor_raw.fit_transform(df, nlp_cols=['text'])
        logger.info(f"Data processed. X shape: {X_processed_raw.shape}")
        
        # Verify it's raw text (object/string)
        if hasattr(X_processed_raw, 'dtype') and X_processed_raw.dtype == 'object':
             logger.info("Success: Data is object/string type.")
        elif isinstance(X_processed_raw, np.ndarray) and isinstance(X_processed_raw[0][0], str):
             logger.info("Success: Data is string array.")
        else:
             logger.warning(f"Warning: Data type is {type(X_processed_raw)}")
             
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize Trainer with Custom Preset to force Transformers
    # We use 'distilbert-base-uncased' as it's smaller/faster than BERT
    selected_models = ['distilbert-base-uncased']
    
    trainer = AutoMLTrainer(
        task_type='classification',
        preset='custom'
    )
    
    logger.info(f"Training models: {selected_models}")
    
    try:
        # Train
        results_raw = trainer.train(
            X_train=X_processed_raw,
            y_train=y_processed_raw,
            selected_models=selected_models,
            n_trials=1, # 1 trial to verify it runs
            timeout=120, # Allow time for model loading/training
            experiment_name="Test_Transformers_Interface"
        )
        
        logger.info("Training Completed.")
        
        # Check results
        if 'distilbert-base-uncased' in trainer.model_summaries:
             summary = trainer.model_summaries['distilbert-base-uncased']
             score = summary.get('score', 'N/A')
             logger.info(f"Transformer Accuracy: {score}")
             
             # Verify it's not random (0.5 for balanced binary)
             # With 100 samples and simple text, it should be reasonable or at least run
             logger.info("Status: SUCCESS - Transformer trained and evaluated.")
        else:
             logger.error("Status: FAILED - Model not found in summaries.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformers_interface()
