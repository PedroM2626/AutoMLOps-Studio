"""
Lightweight worker entry used by multiprocessing spawn to reliably import
and call the real training worker defined in `src.tracking.manager`.

This indirection prevents pickle/import mismatches on Windows (spawn).
"""

def training_worker_entry(config, log_queue, status_queue, pause_event):
    # Import the actual worker function at runtime to avoid import-time side-effects
    from src.tracking.manager import _training_worker
    return _training_worker(config, log_queue, status_queue, pause_event)
