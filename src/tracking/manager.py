"""
Training Job Manager for AutoMLOps Studio
------------------------------------------
Manages concurrent training jobs using multiprocessing.
Each job runs in an isolated subprocess with live log streaming via Queues.
Supports: submit, pause (Windows-safe), resume, cancel, delete.
"""

import multiprocessing
import os
import sys
import time
import uuid
import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Job Status Enum
# ──────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    PAUSED     = "paused"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"

STATUS_ICONS = {
    JobStatus.QUEUED:    "🔵 Queued",
    JobStatus.RUNNING:   "🟢 Running",
    JobStatus.PAUSED:    "🟡 Paused",
    JobStatus.COMPLETED: "✅ Completed",
    JobStatus.FAILED:    "🔴 Failed",
    JobStatus.CANCELLED: "⚫ Cancelled",
}

# ──────────────────────────────────────────────
# TrainingJob Dataclass
# ──────────────────────────────────────────────
@dataclass
class TrainingJob:
    job_id:       str
    name:         str
    config:       dict
    status:       str = JobStatus.QUEUED
    start_time:   float = field(default_factory=time.time)
    end_time:     Optional[float] = None
    best_score:   Optional[float] = None
    mlflow_run_id: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    error_msg:    Optional[str] = None
    # multiprocessing objects (not serialized for display)
    _process:     Any = field(default=None, repr=False)
    _log_queue:   Any = field(default=None, repr=False)     # Queue[str]
    _status_queue: Any = field(default=None, repr=False)    # Queue[dict]
    _pause_event: Any = field(default=None, repr=False)     # multiprocessing.Event
    # Accumulated data (populated by poll_updates)
    logs:         List[str] = field(default_factory=list)
    trials_data:  List[dict] = field(default_factory=list)
    model_summaries: dict = field(default_factory=dict)
    report_data:  dict = field(default_factory=dict)
    target_metric: str = "ACCURACY"

    @property
    def duration_str(self) -> str:
        end = self.end_time or time.time()
        secs = int(end - self.start_time)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def status_label(self) -> str:
        return STATUS_ICONS.get(self.status, self.status)

    def is_active(self) -> bool:
        return self.status in (JobStatus.RUNNING, JobStatus.PAUSED)


# ──────────────────────────────────────────────
# Worker function (runs in subprocess)
# ──────────────────────────────────────────────
def _training_worker(config: dict, log_queue, status_queue, pause_event):
    """
    Runs inside a subprocess. Executes AutoML training and sends
    progress updates back to the parent via Queues.
    """
    # Set up logging to push to queue
    class QueueHandler(logging.Handler):
        def __init__(self, q):
            super().__init__()
            self._q = q
        def emit(self, record):
            try:
                self._q.put(("log", self.format(record)))
            except Exception:
                pass

    root_logger = logging.getLogger()
    q_handler = QueueHandler(log_queue)
    q_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'))
    q_handler.setLevel(logging.INFO)
    root_logger.addHandler(q_handler)
    root_logger.setLevel(logging.INFO)

    try:
        # Set tracking URI from config
        tracking_uri = config.get('mlflow_tracking_uri', 'sqlite:///mlflow.db')
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)

        # Set DagsHub credentials if provided
        dh_user = config.get('dagshub_user')
        dh_pass = config.get('dagshub_token')
        if dh_user and dh_pass:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dh_user
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dh_pass

        # Import engine
        from src.core.processor import AutoMLDataProcessor
        from src.engines.classical import AutoMLTrainer

        task         = config.get('task', 'classification')
        target       = config.get('target')
        date_col     = config.get('date_col')
        train_df     = config.get('train_df')
        test_df      = config.get('test_df')
        preset       = config.get('preset', 'medium')
        n_trials     = config.get('n_trials')
        timeout      = config.get('timeout')
        time_budget  = config.get('time_budget')
        selected_models = config.get('selected_models')
        manual_params   = config.get('manual_params')
        experiment_name = config.get('experiment_name', 'AutoML_Experiment')
        random_state    = config.get('random_state', 42)
        validation_strategy = config.get('validation_strategy', 'auto')
        validation_params   = config.get('validation_params', {})
        ensemble_config     = config.get('ensemble_config', {})
        optimization_mode   = config.get('optimization_mode', 'bayesian')
        optimization_metric = config.get('optimization_metric', 'accuracy')
        nlp_config          = config.get('nlp_config', {})
        selected_nlp_cols   = config.get('selected_nlp_cols', [])
        early_stopping      = config.get('early_stopping', 10)
        forecast_horizon    = config.get('forecast_horizon', 1)
        target_metric_name  = config.get('target_metric_name', 'ACCURACY')
        stability_config    = config.get('stability_config')

        log_queue.put(("log", f"[JOB] Starting preprocessing for experiment: {experiment_name}"))

        # Data Processing
        processor = AutoMLDataProcessor(
            target_column=target, task_type=task,
            date_col=date_col, forecast_horizon=forecast_horizon,
            nlp_config=nlp_config
        )
        X_train_proc, y_train_proc = processor.fit_transform(train_df, nlp_cols=selected_nlp_cols)

        if test_df is not None:
            X_test_proc, y_test_proc = processor.transform(test_df)
        else:
            X_test_proc, y_test_proc = None, None

        feature_names = processor.get_feature_names()
        class_names = None
        if hasattr(processor, 'label_encoder') and processor.label_encoder is not None:
            try:
                class_names = processor.label_encoder.classes_.tolist()
            except Exception:
                pass

        log_queue.put(("log", f"[JOB] Preprocessing done. Features: {len(feature_names) if feature_names else 'N/A'}"))

        # Trials accumulator
        trials_data = []
        report_data = {}
        model_summaries_snapshot = {}

        # -- Callback from trainer (runs in same subprocess, same thread) --
        def callback(trial, score, full_name, dur, metrics=None):
            # Check pause_event
            while pause_event.is_set():
                time.sleep(0.5)

            if metrics and '__report__' in metrics:
                report = metrics['__report__']
                model_name = report.get('model_name', 'Unknown')
                report_data[model_name] = report

                # Serialize plots to buffers
                import io
                serialized_plots = {}
                for pname, pobj in report.get('plots', {}).items():
                    try:
                        import matplotlib
                        from PIL import Image
                        if isinstance(pobj, Image.Image):
                            buf = io.BytesIO()
                            pobj.save(buf, format='PNG')
                            serialized_plots[pname] = ('pil', buf.getvalue())
                        elif hasattr(pobj, 'savefig'):
                            buf = io.BytesIO()
                            pobj.savefig(buf, format='png', bbox_inches='tight', dpi=80)
                            serialized_plots[pname] = ('mpl', buf.getvalue())
                    except Exception:
                        pass
                serialized_report = {k: v for k, v in report.items() if k != 'plots'}
                serialized_report['plots'] = serialized_plots
                status_queue.put({"type": "report", "model_name": model_name, "report": serialized_report})
                return

            try:
                algo_name  = full_name.split(" - ")[0]
                trial_label = full_name.split(" - ")[1]
                trial_num   = int(trial_label.replace("Trial ", ""))
            except Exception:
                algo_name  = full_name
                trial_num  = getattr(trial, 'number', 0)

            import numpy as np
            trial_info = {
                "Global Trial": getattr(trial, 'number', 0) + 1,
                "Model Trial": trial_num,
                "Model": algo_name,
                "Identifier": full_name,
                "Duration (s)": dur,
            }
            if metrics:
                for m_k, m_v in metrics.items():
                    if m_k != "__report__" and isinstance(m_v, (int, float, np.number)):
                        trial_info[m_k.upper()] = m_v
            if target_metric_name not in trial_info:
                trial_info[target_metric_name] = score

            trials_data.append(trial_info)
            status_queue.put({"type": "trial", "trial": trial_info, "score": float(score), "full_name": full_name})

        # Training — forward DL/ensemble flags from job config
        use_ensemble      = config.get('use_ensemble', True)
        use_deep_learning = config.get('use_deep_learning', True)
        ensemble_mode     = config.get('ensemble_mode', 'both')  # 'single', 'ensemble_only', 'both'
        trainer = AutoMLTrainer(
            task_type=task,
            preset=preset,
            ensemble_config=ensemble_config,
            use_ensemble=use_ensemble,
            use_deep_learning=use_deep_learning,
            ensemble_mode=ensemble_mode,
        )
        clean_exp_name = "".join(c for c in experiment_name if ord(c) < 128) or "AutoML_Experiment"

        best_model = trainer.train(
            X_train_proc, y_train_proc,
            n_trials=n_trials,
            timeout=timeout,
            time_budget=time_budget,
            callback=callback,
            selected_models=selected_models,
            early_stopping_rounds=early_stopping,
            manual_params=manual_params,
            experiment_name=clean_exp_name,
            random_state=random_state,
            validation_strategy=validation_strategy,
            validation_params=validation_params,
            optimization_mode=optimization_mode,
            optimization_metric=optimization_metric,
            stability_config=stability_config,
            feature_names=feature_names,
            class_names=class_names,
        )

        best_score = getattr(trainer, 'best_score', None)
        best_params = getattr(trainer, 'best_params', {})
        model_summaries_snapshot = getattr(trainer, 'model_summaries', {})
        consumption_code = getattr(trainer, 'best_consumption_code', None)

        # Get MLflow run ID
        run_id = getattr(trainer, 'best_run_id', None)
        if not run_id:
            try:
                import mlflow
                active = mlflow.active_run()
                if active:
                    run_id = active.info.run_id
            except Exception:
                pass

        # Evaluate on test set
        eval_metrics = None
        if X_test_proc is not None:
            try:
                eval_metrics, _ = trainer.evaluate(X_test_proc, y_test_proc)
            except Exception as e:
                log_queue.put(("log", f"[JOB] Evaluation failed: {e}"))

        log_queue.put(("log", f"[JOB] Training complete! Best: {best_params.get('model_name','?')} Score: {best_score:.4f}" if best_score else "[JOB] Training complete!"))

        # Serialize model summaries (drop non-serializable plot objects)
        safe_summaries = {}
        for m_name, info in model_summaries_snapshot.items():
            safe_info = {k: v for k, v in info.items() if k not in ('model', 'plots')}
            safe_summaries[m_name] = safe_info

        status_queue.put({
            "type": "done",
            "best_score": float(best_score) if best_score is not None else None,
            "best_params": best_params,
            "mlflow_run_id": run_id,
            "mlflow_experiment": clean_exp_name,
            "model_summaries": safe_summaries,
            "eval_metrics": eval_metrics,
            "consumption_code": consumption_code,
        })

    except Exception as e:
        err = traceback.format_exc()
        log_queue.put(("log", f"[JOB ERROR] {e}\n{err}"))
        status_queue.put({"type": "error", "error": str(e)})


# ──────────────────────────────────────────────
# Training Job Manager
# ──────────────────────────────────────────────
class TrainingJobManager:
    """
    Manages a collection of TrainingJob objects.
    Store one instance in st.session_state['job_manager'].
    """

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self._last_poll = 0.0

    # ── Submit ───────────────────────────────
    def submit_job(self, config: dict, name: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())[:8]
        if not name:
            ds = config.get('experiment_name', 'job')
            name = f"{ds}_{job_id}"

        ctx = multiprocessing.get_context("spawn")
        log_queue    = ctx.Queue()
        status_queue = ctx.Queue()
        pause_event  = ctx.Event()   # set = paused

        job = TrainingJob(
            job_id=job_id,
            name=name,
            config=config,
            status=JobStatus.RUNNING,
            start_time=time.time(),
            _log_queue=log_queue,
            _status_queue=status_queue,
            _pause_event=pause_event,
            target_metric=config.get('target_metric_name', 'ACCURACY'),
        )

        process = ctx.Process(
            target=_training_worker,
            args=(config, log_queue, status_queue, pause_event),
            daemon=True,
        )
        process.start()
        job._process = process

        self.jobs[job_id] = job
        logger.info(f"Submitted job {job_id} ({name}) — PID {process.pid}")
        return job_id

    # ── Pause / Resume ───────────────────────
    def pause_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job._pause_event.set()
            job.status = JobStatus.PAUSED
            logger.info(f"Job {job_id} paused.")

    def resume_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job._pause_event.clear()
            job.status = JobStatus.RUNNING
            logger.info(f"Job {job_id} resumed.")

    # ── Cancel ───────────────────────────────
    def cancel_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.is_active():
            if job._process and job._process.is_alive():
                job._process.terminate()
                job._process.join(timeout=5)
            job.status = JobStatus.CANCELLED
            job.end_time = time.time()
            logger.info(f"Job {job_id} cancelled.")

    # ── Delete ───────────────────────────────
    def delete_job(self, job_id: str, delete_mlflow_run: bool = False):
        job = self.jobs.get(job_id)
        if not job:
            return
        # Cancel if still running
        if job.is_active():
            self.cancel_job(job_id)
        # Optionally clean MLflow
        if delete_mlflow_run and job.mlflow_run_id:
            try:
                import mlflow
                mlflow.delete_run(job.mlflow_run_id)
            except Exception as e:
                logger.warning(f"Could not delete MLflow run {job.mlflow_run_id}: {e}")
        del self.jobs[job_id]
        logger.info(f"Job {job_id} deleted.")

    # ── Poll Updates ─────────────────────────
    def poll_updates(self):
        """
        Drain the queues of all active jobs updating their state.
        Call this on every Streamlit rerun.
        Throttled to 0.5s to save CPU.
        """
        now = time.time()
        if now - self._last_poll < 0.5:
            return
        self._last_poll = now

        for job in list(self.jobs.values()):
            if not job.is_active() and job._process is None:
                continue

            # Check if process died unexpectedly
            if job._process and not job._process.is_alive() and job.status == JobStatus.RUNNING:
                job.status = JobStatus.FAILED
                job.end_time = time.time()

            # Drain log queue
            if job._log_queue:
                while True:
                    try:
                        kind, msg = job._log_queue.get_nowait()
                        if kind == "log":
                            job.logs.append(msg)
                            if len(job.logs) > 500:  # cap
                                job.logs = job.logs[-500:]
                    except Exception:
                        break

            # Drain status queue
            if job._status_queue:
                while True:
                    try:
                        update = job._status_queue.get_nowait()
                        utype = update.get("type")

                        if utype == "trial":
                            job.trials_data.append(update["trial"])
                            job.best_score = max(job.best_score or -1e9, update.get("score", -1e9))

                        elif utype == "report":
                            model_name = update["model_name"]
                            job.report_data[model_name] = update["report"]

                        elif utype == "done":
                            job.status = JobStatus.COMPLETED
                            job.end_time = time.time()
                            job.best_score = update.get("best_score") or job.best_score
                            job.mlflow_run_id = update.get("mlflow_run_id")
                            job.mlflow_experiment = update.get("mlflow_experiment")
                            job.model_summaries = update.get("model_summaries", {})
                            job.config['eval_metrics'] = update.get("eval_metrics")
                            job.config['best_params'] = update.get("best_params", {})
                            job.config['consumption_code'] = update.get("consumption_code")

                        elif utype == "error":
                            job.status = JobStatus.FAILED
                            job.end_time = time.time()
                            job.error_msg = update.get("error")

                    except Exception:
                        break

    # ── Helpers ──────────────────────────────
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        return sorted(self.jobs.values(), key=lambda j: j.start_time, reverse=True)

    def has_running_jobs(self) -> bool:
        return any(j.status == JobStatus.RUNNING for j in self.jobs.values())

    def active_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.is_active())
