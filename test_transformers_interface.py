import os
import pandas as pd
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock environment variables to avoid conflicts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current directory to path
sys.path.append(os.getcwd())

logger.info("Pre-importing torch and transformers...")
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    logger.info("Pre-import successful.")
except ImportError as e:
    logger.error(f"Pre-import failed: {e}")

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

    # Create dummy dataset
    data = {
        'text': [
            "This is a positive review.",
            "I hate this product.",
            "Absolutely fantastic service!",
            "Terrible experience, never again.",
            "It was okay, not great.",
            "Highly recommended!"
        ] * 5,
        'sentiment': [1, 0, 1, 0, 0, 1] * 5
    }
    df = pd.DataFrame(data)
    
    logger.info(f"Dataset created with shape: {df.shape}")

    # Initialize Data Processor (TEST 1: Standard TF-IDF)
    # We use standard NLP config as requested by user (TF-IDF)
    logger.info("\n--- TEST 1: Standard TF-IDF (Expected to SKIP training) ---")
    nlp_config_tfidf = {
        'cleaning_mode': 'standard',
        'vectorizer': 'tfidf',
        'max_features': 100, # Small for test
        'ngram_range': (1, 1)
    }
    
    processor_tfidf = AutoMLDataProcessor(
        target_column='sentiment',
        task_type='classification',
        nlp_config=nlp_config_tfidf
    )
    
    # Process Data
    logger.info("Processing data (TF-IDF)...")
    try:
        X_processed_tfidf, y_processed_tfidf = processor_tfidf.fit_transform(df, nlp_cols=['text'])
        logger.info(f"Data processed. X shape: {X_processed_tfidf.shape}")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return

    # Initialize Trainer
    # We explicitly select ONLY transformer models to test them
    selected_models = ['distilbert-base-uncased']
    
    trainer = AutoMLTrainer(
        task_type='classification',
        preset='custom' # Use custom to control models
    )
    
    logger.info(f"Training models (TF-IDF): {selected_models}")
    
    # Train
    try:
        # We use a very short timeout and few trials for the test
        results = trainer.train(
            X_train=X_processed_tfidf,
            y_train=y_processed_tfidf,
            selected_models=selected_models,
            n_trials=1, # Minimal trials
            timeout=30, # 30s max
            experiment_name="Test_Transformers_TFIDF"
        )
        logger.info("Test 1 Completed.")
        if 'distilbert-base-uncased' in trainer.model_summaries:
            logger.info(f"Test 1 Accuracy: {trainer.model_summaries['distilbert-base-uncased'].get('score', 'N/A')}")
        else:
            logger.info("Test 1 Accuracy: Model not run or failed completely.")
            
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    # TEST 2: Raw Text (Passthrough)
    logger.info("\n--- TEST 2: Raw Text Passthrough (Expected to TRAIN/LOAD model) ---")
    
    # Re-instantiate trainer to clear state
    trainer = AutoMLTrainer(
        task_type='classification',
        preset='custom'
    )
    
    nlp_config_raw = {
        'cleaning_mode': 'standard',
        'vectorizer': 'passthrough', # NEW FEATURE
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
        X_processed_raw, y_processed_raw = processor_raw.fit_transform(df, nlp_cols=['text'])
        logger.info(f"Data processed. X shape: {X_processed_raw.shape} (Should be (N, 1))")
        logger.info(f"First element: {X_processed_raw[0]}")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return
        
    logger.info(f"Training models (Raw): {selected_models}")
    try:
        results_raw = trainer.train(
            X_train=X_processed_raw,
            y_train=y_processed_raw,
            selected_models=selected_models,
            n_trials=1,
            timeout=60, # Allow more time for loading
            experiment_name="Test_Transformers_Raw"
        )
        logger.info("Test 2 Completed.")
        
        if 'distilbert-base-uncased' in trainer.model_summaries:
             logger.info(f"Test 2 Accuracy: {trainer.model_summaries['distilbert-base-uncased'].get('score', 'N/A')}")
             logger.info(f"Test 2 Best Params: {trainer.model_summaries['distilbert-base-uncased'].get('params', 'N/A')}")
        else:
             logger.info("Test 2 Accuracy: Model not run or failed completely.")

        if trainer.best_model:
             logger.info(f"Best model object (Raw): {trainer.best_model}")
             
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformers_interface()
