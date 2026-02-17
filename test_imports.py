
print("Starting imports...")
try:
    from automl_engine import AutoMLDataProcessor, AutoMLTrainer, TransformersWrapper
    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
