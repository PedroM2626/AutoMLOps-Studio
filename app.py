from automl_engine import AutoMLDataProcessor, AutoMLTrainer
# from cv_engine import CVAutoMLTrainer, get_cv_explanation # Moved to local scope
import streamlit as st
import pandas as pd
import numpy as np
from mlops_utils import (
    MLFlowTracker, DriftDetector, ModelExplainer, get_model_registry, 
    DataLake, register_model_from_run, get_registered_models, get_all_runs,
    get_model_details, load_registered_model, get_run_details
)
from training_manager import TrainingJobManager, JobStatus
import shap
import joblib # type: ignore
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
from PIL import Image
import uuid
import yaml
import json
import time
import plotly.express as px
import mlflow
import logging

# StreamlitLogHandler is replaced by TrainingJobManager queue-based logging.


# 🎨 Custom Styling
try:
    st.set_page_config(page_title="AutoMLOps Studio", layout="wide")
except st.errors.StreamlitAPIException:
    # This happens if run with 'python app.py' instead of 'streamlit run app.py'
    print("ERROR: This app must be run with Streamlit.")
    print("Please run: streamlit run app.py")
    import sys
    sys.exit(1)

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { border-radius: 8px; border: none; background-color: #4CAF50; color: white; transition: 0.3s; }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.02); }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #4CAF50; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 8px 8px 0 0; padding: 10px 20px; border: 1px solid #e0e0e0; color: #000000 !important; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] p { color: #000000 !important; font-weight: 500; }
    .stTabs [aria-selected="true"] p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

datalake = DataLake()

@st.cache_data(ttl=60)
def get_cached_all_runs():
    return get_all_runs()

@st.cache_data(ttl=60)
def get_cached_registered_models():
    return get_registered_models()

# 📊 Sidebar Metrics & Summary
with st.sidebar:
    # Hybrid Rendering Detection
    if os.environ.get('IS_ELECTRON_APP') == 'true':
        st.markdown("`Desktop Mode`")
        
    st.title("Platform Control")
    st.divider()
    
    # Quick Stats (Cached)
    all_runs_df = get_cached_all_runs()
    reg_models = get_cached_registered_models()
    
    st.markdown("### System Overview")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Experiments", len(all_runs_df) if not all_runs_df.empty else 0)
    with col_s2:
        st.metric("Reg. Models", len(reg_models))
        
    st.divider()
    st.markdown("### Active Dataset")
    if 'df' in st.session_state:
        st.success(f"Rows: {st.session_state['df'].shape[0]}\nCols: {st.session_state['df'].shape[1]}")
    else:
        st.warning("No data loaded")

    st.divider()
    
    # --- DagsHub Integration ---
    with st.expander("DagsHub Integration"):
        st.caption("Connect to your DagsHub repository to save experiments remotely.")
        
        # Try to retrieve settings from environment (.env)
        env_user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        env_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        
        # Try to extract the repository name from the URI if it is DagsHub
        default_repo = ""
        if "dagshub.com" in env_uri:
            try:
                # Exemplo: https://dagshub.com/user/repo.mlflow
                parts = env_uri.split("dagshub.com/")
                if len(parts) > 1:
                    repo_part = parts[1].split(".mlflow")[0]
                    if "/" in repo_part:
                        default_repo = repo_part.split("/")[1]
            except:
                pass

        dh_user = st.text_input("DagsHub Username", value=env_user, key="dh_user_input")
        dh_repo = st.text_input("Repository Name", value=default_repo, key="dh_repo_input")
        dh_token = st.text_input("DagsHub Token (API Key)", value=env_pass, type="password", key="dh_token_input")
        
        col_dh1, col_dh2 = st.columns(2)
        
        with col_dh1:
            if st.button("Connect to DagsHub"):
                if dh_user and dh_repo and dh_token:
                    try:
                        # Configure environment variables for MLflow authentication
                        os.environ["MLFLOW_TRACKING_USERNAME"] = dh_user
                        os.environ["MLFLOW_TRACKING_PASSWORD"] = dh_token
                        
                        # Configure Tracking URI
                        remote_uri = f"https://dagshub.com/{dh_user}/{dh_repo}.mlflow"
                        os.environ["MLFLOW_TRACKING_URI"] = remote_uri # Update env for session persistence
                        mlflow.set_tracking_uri(remote_uri)
                        
                        # Try to list experiments to validate connection
                        try:
                            # Simple connection test
                            mlflow.search_experiments(max_results=1)
                            #st.success(f"✅ Connected: {dh_user}/{dh_repo}")
                            st.session_state['dagshub_connected'] = True
                            st.session_state['mlflow_uri'] = remote_uri
                        except Exception as e:
                            st.error(f"❌ Connection failed: {e}")
                            # Revert to local in case of error
                            local_uri = "sqlite:///mlflow.db"
                            mlflow.set_tracking_uri(local_uri)
                            os.environ["MLFLOW_TRACKING_URI"] = local_uri
                    except Exception as e:
                        st.error(f"Error configuring: {e}")
                else:
                    st.warning("Fill in all fields.")
        
        with col_dh2:
            # Disconnect button only if connected (or if URI points to DagsHub)
            is_dagshub = "dagshub.com" in mlflow.get_tracking_uri()
            if st.button("Disconnect (Revert to Local)", disabled=not is_dagshub):
                local_uri = "sqlite:///mlflow.db"
                mlflow.set_tracking_uri(local_uri)
                os.environ["MLFLOW_TRACKING_URI"] = local_uri
                
                # For security, we clear from os.environ to ensure real disconnection
                if "MLFLOW_TRACKING_USERNAME" in os.environ:
                    del os.environ["MLFLOW_TRACKING_USERNAME"]
                if "MLFLOW_TRACKING_PASSWORD" in os.environ:
                    del os.environ["MLFLOW_TRACKING_PASSWORD"]
                    
                st.session_state['dagshub_connected'] = False
                st.info("🔌 Disconnected. Using local MLflow.")
                st.rerun()

        # Mostrar status atual
        current_uri = mlflow.get_tracking_uri()
        if "dagshub.com" in current_uri:
            st.success(f"🟢 Connected to DagsHub")
            st.caption(f"URI: {current_uri}")
        else:
            st.info("⚪ Using Local MLflow (SQLite)")
    
    # Exibir URI atual
    current_uri = mlflow.get_tracking_uri()
    st.caption(f"Tracking URI: `{current_uri}`")

st.title("🚀 AutoMLOps Studio")
st.markdown("Enterprise-grade Automated Machine Learning & MLOps Platform.")

# Session state initialization
if 'trials_data' not in st.session_state: st.session_state['trials_data'] = []
if 'best_model' not in st.session_state: st.session_state['best_model'] = None
if 'job_manager' not in st.session_state:
    st.session_state['job_manager'] = TrainingJobManager()
if 'mlflow_cache' not in st.session_state:
    st.session_state['mlflow_cache'] = {}

# poll_updates moved to Experiments tab fragment for performance

def prepare_multi_dataset(selected_configs, global_split=None, task_type='classification', date_col=None, target_col=None):
    """
    Loads and splits multiple datasets based on user configurations.
    selected_configs: List of dicts with {'name': str, 'version': str, 'split': float}
    global_split: If provided (0.0 to 1.0), overrides individual split configs.
    task_type: Type of task to determine split strategy (e.g., temporal for time_series).
    date_col: Required for temporal split in time_series.
    target_col: Optional, for stratified split in classification.
    """
    train_dfs = []
    test_dfs = []
    
    for config in selected_configs:
        df_ds = datalake.load_version(config['name'], config['version'])
        
        if global_split is not None:
            split_ratio = global_split
        else:
            split_ratio = config['split'] / 100.0
        
        if split_ratio >= 1.0:
            train_dfs.append(df_ds)
        elif split_ratio <= 0.0:
            test_dfs.append(df_ds)
        else:
            if task_type == 'time_series' and date_col and date_col in df_ds.columns:
                # Temporal split
                df_ds = df_ds.sort_values(by=date_col)
                split_idx = int(len(df_ds) * split_ratio)
                tr = df_ds.iloc[:split_idx]
                te = df_ds.iloc[split_idx:]
            else:
                # Stratified split if target is present and task is classification
                stratify_col = None
                if task_type == 'classification' and target_col and target_col in df_ds.columns:
                    # Check if stratification is possible (enough samples per class)
                    if df_ds[target_col].value_counts().min() > 1:
                        stratify_col = df_ds[target_col]
                
                # Random split (stratified if applicable)
                tr, te = train_test_split(df_ds, train_size=split_ratio, random_state=42, stratify=stratify_col)
            
            train_dfs.append(tr)
            test_dfs.append(te)
            
    full_train = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    full_test = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    return full_train, full_test

# 📑 TAB NAVIGATION (Corrected Indices)
tabs = st.tabs([
    "Data", 
    "AutoML & CV", 
    "Experiments", 
    "Model Registry & Deploy",
    "Monitoring 📉"
])

# --- TAB 0: DATA & DRIFT ---
with tabs[0]:
    st.header("📦 Data Lake & Management")
    
    data_tabs = st.tabs(["Data Management", "Drift Detection"])
    
    with data_tabs[0]:
        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            uploaded_files = st.file_uploader("Upload CSV Data", type="csv", accept_multiple_files=True)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Preview: {uploaded_file.name}", df.head(3))
                    
                    dataset_name = st.text_input(f"Dataset Name for {uploaded_file.name}", uploaded_file.name.replace(".csv", ""), key=f"name_{uploaded_file.name}")
                    if st.button(f"Save {uploaded_file.name} to Data Lake", key=f"save_{uploaded_file.name}"):
                        path = datalake.save_dataset(df, dataset_name)
                        st.success(f"Dataset '{dataset_name}' saved and versioned!")
                        st.session_state['df'] = df # Set as last active
        with col_dl2:
            st.subheader("Explore & Load")
            datasets = datalake.list_datasets()
            selected_ds = st.selectbox("Select Dataset to Load", [""] + datasets)
            if selected_ds:
                versions = datalake.list_versions(selected_ds)
                selected_ver = st.selectbox("Select Version", versions)
                if st.button("Delete Model Version"):
                    from mlops_utils import get_model_registry
                    if get_model_registry().delete_model_version(selected_ds, selected_ver):
                        st.success("Version deleted!")
                        st.rerun()

# --- TAB 4: MLOPS MONITORING ---
with tabs[4]:
    st.header("📉 ML Monitoring & Observability")
    st.markdown("Monitor deployed models for Data Drift, Concept Drift, and assess overall Model Robustness & Stability.")

    mon_tabs = st.tabs(["Production Drift", "Model Robustness & Stability"])
    
    with mon_tabs[0]:
        col_mon1, col_mon2 = st.columns([1, 2])

    with col_mon1:
        st.subheader("1. Reference Data (Baseline)")
        st.info("The Baseline is usually the training dataset stored in your Data Lake.")
        mon_datasets = datalake.list_datasets()
        mon_ref_ds = st.selectbox("Select Baseline Dataset", [""] + mon_datasets, key="mon_ref_ds")
        df_baseline = None
        if mon_ref_ds:
            mon_ref_ver = st.selectbox("Baseline Version", datalake.list_versions(mon_ref_ds), key="mon_ref_ver")
            df_baseline = datalake.load_version(mon_ref_ds, mon_ref_ver)
            st.success(f"Loaded Baseline: {df_baseline.shape[0]} rows")

        st.subheader("2. Production Telemetry")
        st.info("Telemetry data collected from the `/predict` API endpoint.")

        # Load Telemetry Data
        telemetry_path = os.path.join("data_lake", "monitoring", "api_telemetry.csv")
        df_telemetry = None
        if os.path.exists(telemetry_path):
            try:
                df_telemetry = pd.read_csv(telemetry_path)
                st.success(f"Found {len(df_telemetry)} prediction logs.")

                # Filter by timeframe option
                days_filter = st.slider("Analyze last N days", 1, 30, 7)
                if '__timestamp' in df_telemetry.columns:
                    df_telemetry['__timestamp'] = pd.to_datetime(df_telemetry['__timestamp'])
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_filter)
                    df_telemetry = df_telemetry[df_telemetry['__timestamp'] >= cutoff_date]
                    st.caption(f"Filtered to {len(df_telemetry)} recent logs.")
            except Exception as e:
                st.error(f"Error loading telemetry: {e}")
        else:
            st.warning("No telemetry data found. Send requests to the API first.")

    with col_mon2:
        st.subheader("📊 Drift Analysis Matrix")

        if df_baseline is not None and df_telemetry is not None and not df_telemetry.empty:
            if st.button("Calculate Data Drift (Deepchecks)", type="primary"):
                with st.spinner("Analyzing data distributions..."):
                    try:
                        import plotly.express as px
                        
                        # Find overlapping features (ignoring metadata columns)
                        meta_cols = [c for c in df_telemetry.columns if c.startswith("__")]
                        operational_cols = df_telemetry.drop(columns=meta_cols, errors='ignore').columns
                        intersect_cols = df_baseline.select_dtypes(include=np.number).columns.intersection(operational_cols)
                        
                        if len(intersect_cols) == 0:
                            st.error("No matching numeric columns found between Baseline and Telemetry.")
                        else:
                            st.info("Initiating Deepchecks Data Drift Suite...")
                            # Try to import Deepchecks
                            try:
                                from deepchecks.tabular import Dataset
                                from deepchecks.tabular.suites import data_drift
                                import tempfile
                                import os
                                
                                # Prepare Deepchecks datasets
                                # Using only intersecting columns
                                ds_train = Dataset(df_baseline[intersect_cols])
                                ds_test = Dataset(df_telemetry[intersect_cols])
                                
                                # Run Suite
                                suite = data_drift()
                                result = suite.run(train_dataset=ds_train, test_dataset=ds_test)
                                
                                # Extract result (html)
                                st.success("Drift Analysis Complete!")
                                
                                # Save HTML to temporary file and read it back
                                fd, path = tempfile.mkstemp(suffix=".html")
                                try:
                                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                                        result.save_as_html(path)
                                        
                                    with open(path, 'r', encoding='utf-8') as f:
                                        html_report = f.read()
                                        
                                    import streamlit.components.v1 as components
                                    st.markdown("### Deepchecks Interactive Report")
                                    components.html(html_report, height=800, scrolling=True)
                                    
                                finally:
                                    os.remove(path)
                                    
                            except ImportError:
                                st.error("Deepchecks is not installed or failed to import. Falling back to native scipy approach.")
                                # Fallback scipy approach
                                from scipy.stats import ks_2samp
                                drift_results = []
                                drift_count = 0
                                
                                for col in intersect_cols:
                                    stat, p_value = ks_2samp(df_baseline[col].dropna(), df_telemetry[col].dropna())
                                    is_drift = p_value < 0.05 
                                    if is_drift: drift_count += 1
                                    
                                    drift_results.append({
                                        "Feature": col,
                                        "KS-Statistic": float(stat),
                                        "P-Value": float(p_value),
                                        "Status": "🔴 Drift Detected" if is_drift else "🟢 Stable"
                                    })
                                
                                st.write(f"### Results: {drift_count} out of {len(intersect_cols)} features drifting")
                                df_report = pd.DataFrame(drift_results)
                                st.dataframe(df_report.style.applymap(
                                    lambda v: 'color: red' if 'Drift' in str(v) else 'color: green', 
                                    subset=['Status']
                                ), use_container_width=True)
                                
                                with st.expander("View Distribution Plots"):
                                    plot_col = st.selectbox("Select Feature to Plot", [r['Feature'] for r in drift_results])
                                    if plot_col:
                                        fig = px.histogram(
                                            df_baseline, x=plot_col, color_discrete_sequence=['#4CAF50'], 
                                            opacity=0.6, nbins=30, title=f"Feature Drift: {plot_col}"
                                        )
                                        fig.add_histogram(
                                            x=df_telemetry[plot_col], name='Production (Telemetry)', 
                                            marker_color='#FF5722', opacity=0.6
                                        )
                                        fig.update_layout(barmode='overlay')
                                        st.plotly_chart(fig, use_container_width=True)


                    except Exception as e:
                        st.error(f"Error during drift analysis: {e}")
        else:
            st.info("👈 Load both Baseline and Telemetry data to run Drift Analysis.")

    with mon_tabs[1]:
        st.subheader("🛡️ Model Robustness & Stability")
        st.markdown("Run live stability tests on your Registered Models against specific Base Datasets.")
        
        models = get_registered_models()
        if not models:
            st.warning("No models found in Registry. Please register a model from Experiments first.")
        else:
            col_stab1, col_stab2 = st.columns([1, 2])
            
            with col_stab1:
                st.markdown("#### 1. Configuration")
                model_source = st.radio("Model Source", ["Registry", "File Upload"], horizontal=True)
                
                loaded_pipeline = None
                if model_source == "Registry":
                    model_names = [m.name for m in models]
                    selected_model_name = st.selectbox("Registered Model", model_names, key="stab_model_sel")
                    
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    versions = client.search_model_versions(f"name='{selected_model_name}'")
                    version_nums = [v.version for v in versions]
                    selected_version = st.selectbox("Version", version_nums, key="stab_ver_sel")
                    
                    if selected_model_name and selected_version:
                        loaded_pipeline = load_registered_model(selected_model_name, selected_version)
                else:
                    uploaded_model = st.file_uploader("Upload Model (.pkl, .joblib, .onnx)", type=["pkl", "joblib", "onnx"])
                    if uploaded_model:
                        try:
                            if uploaded_model.name.endswith(".onnx"):
                                st.warning("Note: Stability analysis requires retraining (`.fit()`). ONNX models compiled for inference may fail unless wrapped properly.")
                                import onnx
                                loaded_pipeline = onnx.load(uploaded_model)
                                st.success("ONNX Model loaded from file! (Warning: Stability testing might fail)")
                            elif uploaded_model.name.endswith(".joblib"):
                                import joblib
                                loaded_pipeline = joblib.load(uploaded_model)
                                st.success("Joblib Model loaded from file!")
                            elif uploaded_model.name.endswith(".pkl"):
                                import pickle
                                loaded_pipeline = pickle.load(uploaded_model)
                                st.success("Pickle Model loaded from file!")
                        except Exception as e:
                            st.error(f"Failed to load model from file: {e}")
                
                stab_datasets = datalake.list_datasets()
                stab_ref_ds = st.selectbox("Test Dataset (Data Lake)", [""] + stab_datasets, key="stab_ref_ds")
                df_stab_ref = None
                target_col = None
                if stab_ref_ds:
                    stab_ref_ver = st.selectbox("Dataset Version", datalake.list_versions(stab_ref_ds), key="stab_ref_ver")
                    df_stab_ref = datalake.load_version(stab_ref_ds, stab_ref_ver)
                    st.success(f"Dataset Loaded: {df_stab_ref.shape[0]} rows")
                    
                    # Dynamic Target Column Selection
                    target_col = st.selectbox("Target Column Name", options=df_stab_ref.columns.tolist(), index=len(df_stab_ref.columns)-1)
                    
                task_type_sel = st.selectbox("Task Type", ["classification", "regression", "clustering", "time_series", "anomaly_detection"])
                
                # Pre-unwrap actual_model strictly for UI param reading
                actual_model = None
                if loaded_pipeline is not None:
                    actual_model = loaded_pipeline
                    if hasattr(loaded_pipeline, "_model_impl"):
                        if hasattr(loaded_pipeline._model_impl, "sklearn_model"):
                            actual_model = loaded_pipeline._model_impl.sklearn_model
                        elif hasattr(loaded_pipeline._model_impl, "python_model") and hasattr(loaded_pipeline._model_impl.python_model, "pipeline"):
                            actual_model = loaded_pipeline._model_impl.python_model.pipeline
            
            with col_stab2:
                st.markdown("#### 2. Stability Analysis Execution")
                test_types = st.multiselect("Select Stability Tests", [
                    "General Stability Check (Seed & Split)", 
                    "Seed Stability (Initialization)", 
                    "Split Stability (Data Variability)",
                    "Hyperparameter Stability",
                    "Noise Injection Robustness",
                    "Slice Stability (Fairness/Bias)",
                    "Missing Value Robustness",
                    "Calibration Stability",
                    "NLP Text Robustness"
                ], default=["General Stability Check (Seed & Split)"])
                
                # Dynamic inputs based on test type
                test_types_str = " ".join(test_types)
                n_iters = 3  # Default lowered for performance
                noise_level = 0.05
                slice_col = None
                hp_name = None
                hp_vals = None
                
                if any(t in test_types_str for t in ["Seed", "Split", "General", "Noise"]):
                    n_iters = st.slider("Number of Iterations / Splits", 2, 20, 3)
                    
                if "Noise Injection Robustness" in test_types:
                    noise_level = st.slider("Noise Level (Fraction of Std Dev / Flip Prob)", 0.01, 0.50, 0.05, 0.01)
                    
                if "Slice Stability (Fairness/Bias)" in test_types:
                    if df_stab_ref is not None:
                        cat_cols = df_stab_ref.select_dtypes(exclude=[np.number]).columns.tolist()
                        if cat_cols:
                            slice_col = st.selectbox("Select Categorical Feature to Slice (Fairness Test)", cat_cols)
                        else:
                            st.warning("No categorical columns found in the dataset for Slice Stability.")
                            
                if "Hyperparameter Stability" in test_types:
                    if actual_model is not None and hasattr(actual_model, "get_params"):
                        params = list(actual_model.get_params().keys())
                        if params:
                            hp_name = st.selectbox("Select Hyperparameter to vary", params)
                            hp_vals_str = st.text_input("Values to test (comma-separated)", "2, 4, 8")
                            try:
                                # Simple parsing
                                hp_vals = []
                                for v in hp_vals_str.split(","):
                                    v = v.strip()
                                    if v.isdigit(): hp_vals.append(int(v))
                                    elif '.' in v: hp_vals.append(float(v))
                                    elif v.lower() == 'none': hp_vals.append(None)
                                    elif v.lower() == 'true': hp_vals.append(True)
                                    elif v.lower() == 'false': hp_vals.append(False)
                                    else: hp_vals.append(v)
                            except:
                                hp_vals = [v.strip() for v in hp_vals_str.split(",")]
                        else:
                            st.warning("Model does not expose hyperparameters.")
                    else:
                        st.info("Load a model first to view its hyperparameters.")
                    
                if "Calibration Stability" in test_types:
                    st.info("ℹ️ Calibration stability uses Cross-Validation (5 splits) to calculate the Brier Score.")
                
                if df_stab_ref is not None and target_col and target_col in df_stab_ref.columns and loaded_pipeline is not None:
                    if st.button("🚀 Run Stability Analysis", type="primary"):
                        if not test_types:
                            st.warning("Please select at least one stability test to run.")
                        else:
                            with st.spinner("Preparing stability engine..."):
                                try:
                                    # Prepare Model and Data
                                    X_raw = df_stab_ref.drop(columns=[target_col])
                                    y_raw = df_stab_ref[target_col]
                                    
                                    # Process Data using AutoMLDataProcessor
                                    from automl_engine import AutoMLDataProcessor
                                    from sklearn.pipeline import Pipeline
                                    is_pipeline = isinstance(actual_model, Pipeline)
                                    
                                    if is_pipeline:
                                        # If the loaded model is already a Pipeline, it expects raw DataFrame input.
                                        X_stab, y_stab = X_raw, y_raw
                                        processor = None # Override to None since internal transformer handles it
                                    else:
                                        # Use or create an external processor for bare estimators
                                        processor = st.session_state.get('processor')
                                        if not processor:
                                            st.info("⚠️ Active preprocessor not found in session (model loaded from registry). Fitting a temporary encoder for the stability test.")
                                            processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type_sel)
                                            X_stab, y_stab = processor.fit_transform(df_stab_ref)
                                        else:
                                            old_target = processor.target_column
                                            processor.target_column = target_col
                                            X_stab, y_stab = processor.transform(df_stab_ref)
                                            processor.target_column = old_target
                                        
                                    if y_stab is None:
                                        y_stab = y_raw
                                        
                                    from stability_engine import StabilityAnalyzer
                                    analyzer = StabilityAnalyzer(base_model=actual_model, X=X_stab, y=y_stab, task_type=task_type_sel)
                                    
                                    # Execute Tests Sequentially
                                    for t_idx, tt in enumerate(test_types):
                                        st.markdown(f"---")
                                        st.markdown(f"### ⚙️ {tt}")
                                        with st.spinner(f"Running {tt}..."):
                                            if "General Stability" in tt:
                                                report = analyzer.run_general_stability_check(n_iterations=n_iters)
                                                
                                                st.markdown("##### 🎲 Seed Stability (Initialization Variance)")
                                                if not report['seed_stability'].empty:
                                                    st.dataframe(report['seed_stability'][['mean', 'std', 'stability_score']].style.highlight_max(axis=0, subset=['stability_score'], color='lightgreen'))
                                                else:
                                                    st.warning("Seed test yielded empty metrics.")
                                                    
                                                st.markdown("##### 🔀 Split Stability (Data Variance)")
                                                if not report['split_stability'].empty:
                                                    st.dataframe(report['split_stability'][['mean', 'std', 'stability_score']].style.highlight_max(axis=0, subset=['stability_score'], color='lightgreen'))
                                                else:
                                                    st.warning("Split test yielded empty metrics.")
                                                    
                                            elif "Seed Stability" in tt:
                                                raw_res = analyzer.run_seed_stability(n_iterations=n_iters)
                                                agg_res = analyzer.calculate_stability_metrics(raw_res)
                                                st.dataframe(agg_res)
                                                with st.expander("View Raw Iterations Data"):
                                                    st.dataframe(raw_res)
                                                    
                                            elif "Split Stability" in tt:
                                                raw_res = analyzer.run_split_stability(n_splits=n_iters)
                                                agg_res = analyzer.calculate_stability_metrics(raw_res)
                                                st.dataframe(agg_res)
                                                with st.expander("View Raw Iterations Data"):
                                                    st.dataframe(raw_res)
                                                    
                                            elif "Hyperparameter Stability" in tt:
                                                if hp_name and hp_vals:
                                                    raw_res = analyzer.run_hyperparameter_stability(hp_name, hp_vals)
                                                    agg_res = analyzer.calculate_stability_metrics(raw_res)
                                                    st.markdown(f"**Hyperparameter:** `{hp_name}`")
                                                    st.dataframe(agg_res)
                                                    with st.expander("View Raw Iterations Data"):
                                                        st.dataframe(raw_res)
                                                else:
                                                    st.error("Missing hyperparameter configuration.")
                                                    
                                            elif "Noise Injection" in tt:
                                                raw_res = analyzer.run_noise_injection_stability(noise_level=noise_level, n_iterations=n_iters)
                                                agg_res = analyzer.calculate_stability_metrics(raw_res)
                                                st.markdown(f"**Results under {noise_level*100:.1f}% noise**")
                                                st.dataframe(agg_res)
                                                with st.expander("View Raw Iterations Data"):
                                                    st.dataframe(raw_res)
                                                    
                                            elif "Slice Stability" in tt:
                                                if slice_col:
                                                    res = analyzer.run_slice_stability(slice_col)
                                                    st.markdown(f"**Fairness & Slice Stability across `{slice_col}`**")
                                                    st.dataframe(res)
                                                else:
                                                    st.error("Please select a categorical column to test slice stability.")
                                                    
                                            elif "Missing Value" in tt:
                                                res = analyzer.run_missing_value_robustness()
                                                st.markdown("**Imputation Resilience (Performance at different NaN rates)**")
                                                st.dataframe(res)
                                                
                                            elif "Calibration" in tt:
                                                if task_type_sel != "classification":
                                                    st.error("Calibration test is only available for Classification tasks.")
                                                else:
                                                    res = analyzer.run_calibration_stability(n_splits=5)
                                                    st.markdown("**Cross-Validated Brier Scores (Lower is better)**")
                                                    if 'error' in res.columns:
                                                        st.error(res.iloc[0]['error'])
                                                    else:
                                                        metrics = analyzer.calculate_stability_metrics(res)
                                                        st.dataframe(metrics)
                                                        with st.expander("View Splits"):
                                                            st.dataframe(res)
                                                            
                                            elif "NLP Text" in tt:
                                                tf_func = processor.transform if processor else None
                                                if tf_func:
                                                    # Need to temporarily unset target_column on processor so it doesn't fail parsing raw X
                                                    old_tgt = processor.target_column
                                                    processor.target_column = None
                                                
                                                res = analyzer.run_nlp_robustness(
                                                    n_iterations=n_iters, 
                                                    typo_probability=0.1,
                                                    X_raw=X_raw,
                                                    transform_func=tf_func
                                                )
                                                
                                                if tf_func:
                                                    processor.target_column = old_tgt
                                                    
                                                st.markdown("**NLP Robustness (Text Injection/Typos)**")
                                                if 'error' in res.columns:
                                                    st.error(res.iloc[0]['error'])
                                                else:
                                                    metrics = analyzer.calculate_stability_metrics(res)
                                                    st.dataframe(metrics)
                                                    with st.expander("View Iterations"):
                                                        st.dataframe(res)
                                            
                                    st.success("Analysis Complete!")
                                    
                                except Exception as e:
                                    import traceback
                                    err_trace = traceback.format_exc()
                                    st.error(f"Error executing Stability Analysis: {e}")
                                    with open('error_trace.txt', 'w') as f:
                                        f.write(err_trace)
                elif df_stab_ref is not None:
                    st.error(f"Target column '{target_col}' not found in the selected dataset.")
                else:
                    st.info("Wait for dataset selection to enable analysis.")
    with data_tabs[1]:
        st.subheader("📉 Data Drift Analysis")
        st.markdown("Compare two datasets (e.g., Training vs Inference) to detect distribution shifts.")

        # Select Reference Dataset (Training)
        st.markdown("#### 1. Reference Data (Baseline)")
        ref_ds = st.selectbox("Reference Dataset", [""] + datasets, key="drift_ref_ds")
        df_ref = None
        if ref_ds:
            ref_ver = st.selectbox("Reference Version", datalake.list_versions(ref_ds), key="drift_ref_ver")
            df_ref = datalake.load_version(ref_ds, ref_ver)
            st.write(f"Reference Loaded: {df_ref.shape}")

        # Select Current Dataset (Production/New)
        st.markdown("#### 2. Current Data (Target)")
        curr_ds = st.selectbox("Current Dataset", [""] + datasets, key="drift_curr_ds")
        df_curr = None
        if curr_curr := curr_ds: # walrus operator just to match style
            curr_ver = st.selectbox("Current Version", datalake.list_versions(curr_ds), key="drift_curr_ver")
            df_curr = datalake.load_version(curr_ds, curr_ver)
            st.write(f"Current Loaded: {df_curr.shape}")
            
        if df_ref is not None and df_curr is not None:
            if st.button("🚀 Run Drift Detection"):
                with st.spinner("Calculating Drift Metrics (PSI, KS)..."):
                    # Basic Drift Logic (Placeholder for sophisticated library like evidently)
                    drift_report = []
                    numeric_cols = df_ref.select_dtypes(include=np.number).columns.intersection(df_curr.columns)
                    
                    for col in numeric_cols:
                        # KS Test
                        from scipy.stats import ks_2samp
                        stat, p_value = ks_2samp(df_ref[col].dropna(), df_curr[col].dropna())
                        drift_detected = p_value < 0.05
                        drift_report.append({
                            "Feature": col,
                            "KS Stat": stat,
                            "P-Value": p_value,
                            "Drift Detected": "🔴 YES" if drift_detected else "🟢 NO"
                        })
                        
                        # Visualization
                        fig = px.histogram(df_ref, x=col, color_discrete_sequence=['blue'], opacity=0.5, nbins=30, title=f"Distribution: {col}")
                        fig.add_histogram(x=df_curr[col], name='Current', marker_color='red', opacity=0.5)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(pd.DataFrame(drift_report))


# --- TAB 1: AUTOML & MODEL HUB ---
with tabs[1]:
    st.header("🤖 AutoML & Model Hub")
    
    # Sub-tabs within AutoML
    automl_tabs = st.tabs(["📊 Classical ML (Tabular)", "🖼️ Computer Vision"])
    
    # --- SUB-TAB 1.1: CLASSICAL ML ---
    with automl_tabs[0]:
        st.subheader("📋 Training Configuration (Tabular)")
        
        # 1. Definição da Tarefa
        col_t0, col_t1, col_t2 = st.columns([1, 1, 1])
        with col_t0:
            learning_type = st.radio("Learning Type", ["Supervised", "Unsupervised"], key="learning_type_selector")

        with col_t1:
            if learning_type == "Supervised":
                task_options = ["classification", "regression", "time_series"]
            else:
                task_options = ["clustering", "anomaly_detection", "dimensionality_reduction"]
                
            task = st.selectbox("Task Type", task_options, key="task_selector_train")
        
        with col_t2:
            training_strategy = st.radio("Hyperparameter Configuration", ["Automatic", "Manual"], 
                                         help="Automatic: System finds best parameters. Manual: You explicitly define them.")

        st.divider()

        # 2. Configuração de Modelos e Parâmetros
        st.subheader("🎯 Model Selection")
        
        # Seletor de Fonte do Modelo (Migrado de Fine-Tune)
        model_source = st.radio("Model Source", 
                               ["Standard AutoML (Scikit-Learn/XGBoost/Transformers)", 
                                "Model Registry (Registered)", 
                                "Local Upload (.pkl)"],
                               horizontal=True)

        trainer_temp = AutoMLTrainer(task_type=task)
        available_models = trainer_temp.get_available_models()
        
        selected_models = None
        manual_params = None
        
        # Lógica de Seleção Baseada na Fonte
        ensemble_config = {} # Initialize empty ensemble config

        if model_source == "Standard AutoML (Scikit-Learn/XGBoost/Transformers)":
            mode_selection = st.radio("Model Selection", ["Automatic (Preset)", "Manual (Select)", "Custom Ensemble Builder"], horizontal=True)
            
            if mode_selection == "Manual (Select)":
                selected_models = st.multiselect("Choose Models", available_models, default=available_models[:2] if available_models else None)
                
            elif mode_selection == "Custom Ensemble Builder":
                st.markdown("##### 🏗️ Custom Ensemble Builder")
                st.info("Create an ensemble by combining multiple base models. The system will train the final ensemble.")
                
                ensemble_type = st.selectbox("Ensemble Type", ["Voting", "Stacking"])
                
                # Filter base models (exclude other ensembles/custom models to avoid recursion for now)
                base_candidates = [m for m in available_models if 'ensemble' not in m and 'custom' not in m]
                
                st.markdown("**1. Select Base Estimators**")
                selected_base_models = st.multiselect(
                    "Base Estimators (Components)", 
                    base_candidates, 
                    default=base_candidates[:3] if len(base_candidates) > 3 else base_candidates
                )
                
                if len(selected_base_models) < 2:
                    st.warning("⚠️ Select at least 2 models to form a robust ensemble.")
                
                if ensemble_type == "Voting":
                    st.markdown("**2. Voting Configuration**")
                    if task == "classification":
                        voting_type = st.selectbox("Voting Type", ["soft", "hard"], help="Soft: Average probabilities. Hard: Majority class voting.")
                    else:
                        voting_type = 'soft' # Not used in regressor but safe to keep
                        st.caption("Regression always uses averaged predictions.")
                    
                    use_weights = st.checkbox("Define Weights (Weighted Voting)", help="Allows assigning different weights for each model in voting.")
                    voting_weights = None

                
                if use_weights:
                    st.caption("Enter weights separated by comma in the same order as selected models.")
                    weights_input = st.text_input("Weights (e.g.: 1.0, 2.0)", value=",".join(["1.0"] * len(selected_base_models)))
                    try:
                        voting_weights = [float(w.strip()) for w in weights_input.split(',')]
                        if len(voting_weights) != len(selected_base_models):
                            st.error(f"⚠️ Number of weights ({len(voting_weights)}) differs from number of models ({len(selected_base_models)}). Using equal weights.")
                            voting_weights = None
                    except:
                        st.error("⚠️ Invalid format. Use comma separated numbers.")
                        voting_weights = None

                ensemble_config = {
                    'voting_estimators': selected_base_models,
                    'voting_type': voting_type,
                    'voting_weights': voting_weights
                }
                selected_models = ['custom_voting']
                
                if ensemble_type == "Stacking":
                    st.markdown("**2. Stacking Configuration**")
                    st.info("Stacking trains a 'Meta-Model' to learn the best combination of base models.")
                    
                    # Final estimator selection
                    meta_candidates = ['logistic_regression', 'random_forest', 'xgboost', 'linear_regression', 'ridge']
                    # Filter by task
                    if task == 'classification':
                        meta_candidates = [m for m in meta_candidates if m in base_candidates and m != 'linear_regression' and m != 'ridge']
                        if not meta_candidates: meta_candidates = ['logistic_regression']
                    else:
                        meta_candidates = [m for m in meta_candidates if m in base_candidates and m != 'logistic_regression']
                        if not meta_candidates: meta_candidates = ['linear_regression']

                    
                    st.caption(f"Meta-Modelo selecionado: {final_est_name}")
                    
                    ensemble_config = {
                        'stacking_estimators': selected_base_models,
                        'stacking_final_estimator': final_est_name
                    }
                    selected_models = ['custom_stacking']

        elif model_source == "Model Registry (Registered)":
            reg_models = get_registered_models()
            if reg_models:
                base_model_name = st.selectbox("Select Registered Model", [m.name for m in reg_models], key="reg_sel_train")
                selected_models = [base_model_name]
                st.info(f"The model '{base_model_name}' will be used as a base for retraining/fine-tuning.")
            else:
                st.warning("No registered models found.")

        elif model_source == "Local Upload (.pkl)":
            uploaded_pkl = st.file_uploader("Upload base .pkl file", type="pkl", key="pkl_upload_train")
            if uploaded_pkl:
                selected_models = ["Uploaded_Model"] # Placeholder, needs custom backend logic
                st.info("Model loaded for retraining.")

        st.subheader("🎯 Optimization Configuration")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            # Hyperparameter Optimization Mode Selector
            optimization_mode = st.selectbox(
                "Hyperparameter Optimization Mode",
                ["Bayesian Optimization (Default)", "Random Search", "Grid Search", "Hyperband"],
                index=0,
                help="Bayesian: More efficient. Random: Exploratory. Grid: Exhaustive (slow). Hyperband: Fast for many data."
            )
            
            # Mapping to backend
            opt_mode_map = {
                "Bayesian Optimization (Default)": "bayesian",
                "Random Search": "random",
                "Grid Search": "grid",
                "Hyperband": "hyperband"
            }
            selected_opt_mode = opt_mode_map[optimization_mode]

            # Unified preset selector (including 'custom' and 'test')
            if model_source == "Standard AutoML (Scikit-Learn/XGBoost/Transformers)":
                training_preset = st.select_slider(
                    "Training Mode (Preset)",
                    options=["test", "fast", "medium", "high", "custom"],
                    value="medium",
                    help="test: Fast test (1 trial). fast: Fast. medium: Balanced. high: Exhaustive. custom: Define your rules."
                )
            else:
                # For other modes, we allow customization but start with medium
                st.info(f"Base mode adapted for {model_source}")
                # Here we can also allow custom n_trials
                use_custom_tuning = st.checkbox("Customize Optimization (Trials/Timeout)", value=False)
                training_preset = "custom" if use_custom_tuning else "medium"

            # Conditional inputs for custom mode
            if training_preset == "custom":
                st.markdown("##### 🛠️ Custom Configuration")
                n_trials = st.number_input("Number of Trials (per model)", 1, 1000, 20, key="cust_trials")
                timeout_per_model = st.number_input("Timeout per model (seconds)", 10, 7200, 600, key="cust_timeout")
                total_time_budget = st.number_input("Max Total Time (seconds)", 60, 86400, 3600, key="cust_total_time", help="Max time to run the ENTIRE experiment. If exceeded, training stops after the current model.")
                early_stopping = st.number_input("Early Stopping (Rounds)", 0, 50, 7, key="cust_es")
                
                st.markdown("##### ⚡ Advanced Parameters")
                custom_max_iter = st.number_input("Max Iterations (max_iter)", 100, 100000, 1000, help="Iteration limit for solvers (LogisticRegression, SVM, MLP). Very high values can slow down training.")
                
                manual_params = {
                    'max_iter': custom_max_iter
                }
            elif training_preset == "test":
                 st.warning("⚠️ TEST MODE: Running with only 1 trial and short timeout for pipeline validation.")
                 n_trials = 1
                 timeout_per_model = 30
                 total_time_budget = 60
                 early_stopping = 1
                 manual_params = {}
            else:
                n_trials = None
                timeout_per_model = None
                total_time_budget = None
                early_stopping = 10
                manual_params = {}

        
        with col_opt2:
            st.markdown("##### 🛡️ Validation Strategy")
            validation_options = ["Automatic (Recommended)", "K-Fold Cross Validation", "Stratified K-Fold", "Holdout (Train/Test)", "Auto-Split (Optimized)", "Time Series Split"]
            
            # Filter options based on task
            if task == "time_series":
                val_strategy_ui = "Time Series Split"
                st.info("Time series must use temporal splitting.")
                validation_strategy = 'time_series_cv'
            elif task == "classification":
                val_strategy_ui = st.selectbox("Validation Method", validation_options, index=0)
            else: # regression, clustering, anomaly
                # Stratified only makes sense for classification
                opts = [o for o in validation_options if o != "Stratified K-Fold"]
                val_strategy_ui = st.selectbox("Validation Method", opts, index=0)
            
            validation_params = {}
            if val_strategy_ui == "Automatic (Recommended)":
                validation_strategy = 'auto'
                st.info("System will choose the best strategy based on data size.")
            elif val_strategy_ui in ["K-Fold Cross Validation", "Stratified K-Fold"]:
                n_folds = st.number_input("Number of Folds", 2, 20, 5, key="val_folds")
                validation_params['folds'] = n_folds
                validation_strategy = 'cv' if val_strategy_ui == "K-Fold Cross Validation" else 'stratified_cv'
            elif val_strategy_ui == "Holdout (Train/Test)":
                test_size = st.slider("Test Size (%)", 10, 50, 20, key="val_holdout", help="Percentage of training dataset reserved for Internal Validation during optimization (do not confuse with Final Test).") / 100.0
                validation_params['test_size'] = test_size
                validation_strategy = 'holdout'
            elif val_strategy_ui == "Auto-Split (Optimized)":
                st.info("System will decide the best split during optimization.")
                validation_strategy = 'auto_split'
            elif val_strategy_ui == "Time Series Split":
                n_splits = st.number_input("Number of Temporal Splits", 2, 20, 5, key="val_ts_splits")
                validation_params['folds'] = n_splits
                validation_strategy = 'time_series_cv'
            
            # NLP columns selection
            st.markdown("##### 🔤 NLP Configuration")
            
            # Advanced NLP Configurations
            # We use a container to render NLP options later,
            # once we have access to the sample_df (data preview).
            nlp_container = st.container()
            nlp_config_automl = {} 

            if task == "time_series":
                st.info("💡 Temporal split is mandatory for time series.")

        # Novo Seletor de Métrica Alvo (Optimization Metric)
        metric_options = {
            'classification': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
            'regression': ['r2', 'rmse', 'mae'],
            'clustering': ['silhouette'],
            'time_series': ['rmse', 'mae', 'mape'],
            'anomaly_detection': ['f1'],
            'dimensionality_reduction': ['explained_variance']
        }
        
        target_metric_options = metric_options.get(task, ['accuracy'])
        optimization_metric = st.selectbox("Target Metric (Optimization)", target_metric_options, index=0, help="Metric that AutoML will try to maximize (or minimize, depending on the metric).")
        target_metric_name = optimization_metric.upper()

        st.divider()
        st.subheader("🌱 Reproducibility Configuration (Seed)")
        seed_mode = st.radio("Seed Mode", 
                             ["Automatic (Different per model)", 
                              "Automatic (Same for all)", 
                              "Manual (Same for all)", 
                              "Manual (Different per model)"], 
                             horizontal=True)
        
        random_seed_config = 42 # Default
        
        effective_models = selected_models if selected_models else available_models
        
        if seed_mode == "Automatic (Different per model)":
            random_seed_config = {m: np.random.randint(0, 999999) for m in effective_models}
            st.info("🎲 Random seeds will be generated for each model.")
        elif seed_mode == "Automatic (Same for all)":
            random_seed_config = np.random.randint(0, 999999)
            st.info(f"🎲 A single random seed will be used for all: {random_seed_config}")
        elif seed_mode == "Manual (Same for all)":
            random_seed_config = st.number_input("🌱 Enter Global Seed", 0, 999999, 42)
        elif seed_mode == "Manual (Different per model)":
            st.markdown("##### Enter the Seed for each model:")
            random_seed_config = {}
            cols_seed = st.columns(min(len(effective_models), 3))
            for i, m in enumerate(effective_models):
                with cols_seed[i % 3]:
                    random_seed_config[m] = st.number_input(f"Seed: {m}", 0, 999999, 42, key=f"seed_{m}")

        # Hiperparâmetros Manuais integrados nas opções de tuning
        if training_strategy == "Manual":
            st.divider()
            st.subheader("⚙️ Manual Hyperparameter Configuration")
            st.info("Note: In Manual mode, you define the parameters used as a starting point (enqueue) for the selected models.")
            
            # Se múltiplos modelos estiverem selecionados, o usuário pode configurar um por um ou um modelo de referência
            ref_model = st.selectbox("Model to Configure", selected_models or available_models)
            
            # Merge existing manual_params with new manual config
            current_manual_params = manual_params.copy()
            current_manual_params['model_name'] = ref_model
            
            schema = trainer_temp.get_model_params_schema(ref_model)

            if schema:
                st.markdown(f"**Parameters for {ref_model}**")
                cols_p = st.columns(3)
                for i, (p_name, p_config) in enumerate(schema.items()):
                    with cols_p[i % 3]:
                        if p_config[0] == 'int':
                            manual_params[p_name] = st.number_input(p_name, p_config[1], p_config[2], p_config[3])
                        elif p_config[0] == 'float':
                            manual_params[p_name] = st.number_input(p_name, p_config[1], p_config[2], p_config[3], format="%.4f")
                        elif p_config[0] == 'list':
                            options, p_def = p_config[1], p_config[2]
                            manual_params[p_name] = st.selectbox(p_name, options, index=options.index(p_def) if p_def in options else 0)
        else:
            manual_params = None

        st.divider()

        # 3. Seleção de Dados
        st.subheader("📂 Data Selection")
        available_datasets = datalake.list_datasets()
        selected_ds_list = st.multiselect("Choose Datasets", available_datasets, key="ds_train_multi")
        
        target_pre = None
        date_col_pre = None
        sample_df = None
        
        if selected_ds_list:
            try:
                first_ds = selected_ds_list[0]
                versions = datalake.list_versions(first_ds)
                if versions:
                    first_ver = versions[0]
                    sample_df = datalake.load_version(first_ds, first_ver, nrows=5)
                    
                    col_sel1, col_sel2 = st.columns(2)
                    with col_sel1:
                        if task not in ["clustering", "anomaly_detection", "dimensionality_reduction"]:
                            target_pre = st.selectbox("🎯 Target Variable", sample_df.columns, key="target_selector_pre")
                    
                    with col_sel2:
                        if task == "time_series":
                            date_col_pre = st.selectbox("📅 Date Column (REQUIRED)", sample_df.columns, key="ts_date_selector")
            except Exception as e: st.error(f"Error loading sample: {e}")

        selected_configs = []
        if selected_ds_list:
            cols_ds = st.columns(len(selected_ds_list))
            for i, ds_name in enumerate(selected_ds_list):
                with cols_ds[i]:
                    st.markdown(f"**{ds_name}**")
                    versions = datalake.list_versions(ds_name)
                    ver = st.selectbox(f"Version", versions, key=f"ver_{ds_name}")
                    
                    # Configuração de Papel do Dataset (Granularidade Solicitada)
                    if validation_strategy == 'holdout':
                        st.caption("Define dataset role:")
                        role = st.radio("Role", ["Train + Test (Split)", "Train Only (100%)", "Test Only (100%)"], key=f"role_{ds_name}", help="Final destination of data. 'Test Only' reserves data for final evaluation. 'Train' goes to the training pool.")
                        
                        split = 100
                        if role == "Train + Test (Split)":
                            split = st.slider(f"% Train", 10, 95, 80, key=f"split_{ds_name}", help="Percentage of this dataset going to the Training pool. The rest goes to Final Test.")
                        elif role == "Test Only (100%)":
                            split = 0
                    else:
                        # Para estratégias como K-Fold ou Auto-Split, usamos o dataset integralmente no processo (split=100)
                        # O sistema de validação cuidará da divisão interna.
                        split = 100
                        st.info(f"Dataset used entirely for {validation_strategy}")
                    
                    selected_configs.append({'name': ds_name, 'version': ver, 'split': split})

        # Preencher o container de NLP agora que temos acesso aos dados (sample ou train)
        selected_nlp_cols = []
        with nlp_container:
            potential_nlp_cols = []
            if sample_df is not None:
                potential_nlp_cols = sample_df.select_dtypes(include=['object']).columns.tolist()
            elif 'train_df' in st.session_state:
                potential_nlp_cols = st.session_state['train_df'].select_dtypes(include=['object']).columns.tolist()
            
            if potential_nlp_cols:
                selected_nlp_cols = st.multiselect("Text Columns (NLP)", potential_nlp_cols, help="Select the columns containing text for optimized NLP processing.")
                
                if selected_nlp_cols:
                    col_nlp1, col_nlp2 = st.columns(2)
                    with col_nlp1:
                        vectorizer_automl = st.selectbox("Vectorization", ["tfidf", "count", "embeddings"], key="automl_vect")
                        if vectorizer_automl == "embeddings":
                            embedding_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"], index=0)
                        else:
                            ngram_min_automl, ngram_max_automl = st.slider("N-Grams Range", 1, 3, (1, 2), key="automl_ngram")
                    with col_nlp2:
                        if vectorizer_automl != "embeddings":
                            remove_stopwords_automl = st.checkbox("Remove Stopwords (English)", value=True, key="automl_stop")
                            lematization_automl = st.checkbox("Lemmatization (WordNet - requires NLTK)", value=False, key="automl_lemma")
                            max_features_automl = st.number_input("Max Features", min_value=100, max_value=None, value=5000, step=1000, key="automl_max_feat", help="Leave high (e.g., 5000+) to capture more vocabulary. Automatically optimized.")
                        else:
                            st.info("💡 Embeddings generate dense fixed vectors (e.g., 384 dimensions).")

                    nlp_config_automl = {
                        "vectorizer": vectorizer_automl,
                        "embedding_model": embedding_model if vectorizer_automl == "embeddings" else None,
                        "ngram_range": (ngram_min_automl, ngram_max_automl) if vectorizer_automl != "embeddings" else (1, 1),
                        "stop_words": remove_stopwords_automl if vectorizer_automl != "embeddings" else False,
                        "max_features": max_features_automl if vectorizer_automl != "embeddings" else 5000,
                        "lemmatization": lematization_automl if vectorizer_automl != "embeddings" else False
                    }
            else:
                if selected_ds_list:
                    st.info("No text column identified in the sample.")
                else:
                    st.info("Select a dataset below to configure NLP.")

        if selected_configs:
            if st.button("📥 Load and Prepare Data", key="btn_load_train"):
                # Usar configurações individuais de split (global_split=None)
                train_df, test_df = prepare_multi_dataset(selected_configs, global_split=None, task_type=task, date_col=date_col_pre, target_col=target_pre)
                
                st.session_state['train_df'] = train_df
                st.session_state['test_df'] = test_df
                st.session_state['current_task'] = task
                st.session_state['date_col_active'] = date_col_pre
                st.session_state['target_active'] = target_pre # Salvar target selecionado
                st.session_state['n_trials_active'] = n_trials
                st.session_state['early_stopping_active'] = early_stopping
                st.success("Data loaded!")

        if 'train_df' in st.session_state and st.session_state.get('current_task') == task:
            train_df = st.session_state['train_df']
            test_df = st.session_state['test_df']
            
            st.divider()
            st.subheader("⚙️ Final Configuration")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if task not in ["clustering", "anomaly_detection", "dimensionality_reduction"]:
                    # Se já foi selecionado no pré-carregamento, apenas exibir e travar
                    if st.session_state.get('target_active') and st.session_state['target_active'] in train_df.columns:
                        target = st.session_state['target_active']
                        st.info(f"🎯 Target Defined: **{target}** (To change, reload data)")
                    else:
                        target = st.selectbox("🎯 Select Target", train_df.columns)
                else:
                    target = None
            
            with col_f2:
                if task == "time_series":
                    freq = st.selectbox("⏱️ Interval", ["Minutes", "Hours", "Days", "Weeks", "Months", "Years"])
                    forecast_horizon = st.number_input("🔮 Horizon", 1, 100, 7)
                else: forecast_horizon, freq = 1, "D"

            # --- Stability Analysis Integration in AutoML Flow ---
            st.divider()
            st.subheader("⚖️ Stability & Robustness Analysis (Optional)")
            enable_stability = st.checkbox("Run Post-Training Stability Analysis", help="Executes additional robustness tests after AutoML finishes.")
            
            selected_stability_tests = []
            if enable_stability:
                stability_options = [
                    "Data Variation Robustness", 
                    "Initialization Robustness", 
                    "Hyperparameter Sensitivity", 
                    "General Analysis"
                ]
                selected_stability_tests = st.multiselect(
                    "Select Stability Tests", 
                    stability_options,
                    default=["General Analysis"],
                    help="Select which analyses to run automatically on the best found model."
                )
                st.info("📊 Results will be saved to MLflow and a PDF report will be generated.")

            if st.button("🚀 Submit Experiment", key="btn_start_train", type="primary"):
                if not selected_configs:
                    st.error("Please select and load a dataset first.")
                elif 'train_df' not in st.session_state:
                    st.error("Please load the data first (click '📥 Load and Prepare Data').")
                else:
                    exp_tag = selected_configs[0]['name'] if selected_configs else "AutoML"
                    experiment_name = f"{exp_tag}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"
                    clean_experiment_name = "".join(c for c in experiment_name if ord(c) < 128) or "AutoML_Experiment"
                    target_metric_name_local = optimization_metric.upper()

                    job_config = {
                        # Data
                        'train_df': st.session_state.get('train_df'),
                        'test_df': st.session_state.get('test_df'),
                        'target': target,
                        'task': task,
                        'date_col': date_col_pre,
                        'forecast_horizon': forecast_horizon,
                        'selected_nlp_cols': selected_nlp_cols,
                        'nlp_config': nlp_config_automl,
                        # Training
                        'preset': training_preset,
                        'n_trials': n_trials,
                        'timeout': timeout_per_model,
                        'time_budget': total_time_budget,
                        'selected_models': selected_models,
                        'manual_params': manual_params,
                        'experiment_name': clean_experiment_name,
                        'random_state': random_seed_config,
                        'validation_strategy': validation_strategy,
                        'validation_params': validation_params,
                        'ensemble_config': ensemble_config,
                        'optimization_mode': selected_opt_mode,
                        'optimization_metric': optimization_metric,
                        'target_metric_name': target_metric_name_local,
                        'early_stopping': early_stopping,
                        'stability_config': {'tests': selected_stability_tests, 'n_iterations': 3} if enable_stability else None,
                        # MLflow
                        'mlflow_tracking_uri': mlflow.get_tracking_uri(),
                        'dagshub_user': os.environ.get('MLFLOW_TRACKING_USERNAME'),
                        'dagshub_token': os.environ.get('MLFLOW_TRACKING_PASSWORD'),
                    }

                    jm = st.session_state['job_manager']
                    job_id = jm.submit_job(job_config, name=clean_experiment_name)
                    st.success(f"✅ Experiment **{clean_experiment_name}** submitted! (Job ID: `{job_id}`)")
                    st.info("📊 View live progress, logs and results in the **Experiments** tab.")
                    st.balloons()

    # --- SUB-TAB 1.2: COMPUTER VISION ---
    with automl_tabs[1]:
        st.subheader("Computer Vision AutoML")
        cv_task = st.selectbox("CV Task", ["image_classification", "image_segmentation", "object_detection"], key="cv_task_selector")
        
        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            st.markdown("##### Dataset Upload")
            # data_dir input removed in favor of upload
            cv_upload = st.file_uploader("Upload Dataset (ZIP)", type="zip", key="cv_zip_upload", help="Upload a ZIP file containing your images (and labels/masks if applicable). Structure: root/class_name/image.jpg")
            
            data_dir = None
            if cv_upload:
                import zipfile
                import shutil
                
                # Create temp dir for extraction
                temp_extract_dir = "temp_cv_dataset"
                
                def remove_readonly(func, path, excinfo):
                    """Error handler for shutil.rmtree to handle read-only files on Windows."""
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                if os.path.exists(temp_extract_dir):
                    try:
                        shutil.rmtree(temp_extract_dir, onerror=remove_readonly)
                    except Exception as e:
                        st.warning(f"Could not fully clear temporary directory: {e}. Attempting to continue...")
                
                os.makedirs(temp_extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(cv_upload, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                st.success(f"Dataset extracted to temporary folder: {temp_extract_dir}")
                data_dir = temp_extract_dir
                
                # Attempt to detect structure
                subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                if len(subdirs) == 1:
                    # Possibly nested in a root folder
                    data_dir = os.path.join(data_dir, subdirs[0])
                    st.info(f"Detected nested root folder: {subdirs[0]}")
            
            mask_dir = None
            if cv_task == "image_segmentation":
                st.info("For Segmentation, ensure masks are in a 'masks' folder inside the zip or upload separate mask zip.")
                # Simplification: Assume masks are inside the main zip or ask for separate upload if critical
                # mask_dir = st.text_input("Masks Directory (for Segmentation)", "data/images/masks", key="cv_mask_dir")
            elif cv_task == "object_detection":
                st.info("For Detection, ensure annotations are in the zip (COCO/YOLO format).")
                # mask_dir = st.text_input("Annotations Directory (for Detection)", "data/images/annotations", key="cv_annot_dir")
            
                
        with col_cv2:
            epochs = st.number_input("Epochs", 1, 100, 5, key="cv_epochs")
            lr_cv = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", key="cv_lr")

        if st.button("🚀 Start CV Training", key="cv_start_btn"):
            if not data_dir:
                st.error("Please upload a dataset ZIP file first.")
            else:
                from cv_engine import CVAutoMLTrainer
                
                trainer = CVAutoMLTrainer(task_type=cv_task)
                
                status_cv = st.empty()
                progress_cv = st.progress(0)
                
                def cv_callback(epoch, acc, loss, duration):
                    status_cv.write(f"Epoch {epoch}: Acc={acc:.4f}, Loss={loss:.4f}, Time={duration:.2f}s")
                    progress_cv.progress((epoch + 1) / epochs)

                with st.spinner("Training vision model..."):
                    try:
                        # Container for real-time logs in CV
                        st.markdown("### 🖥️ CV Execution Logs (Real-time)")
                        cv_log_expander = st.expander("View Training Logs", expanded=True)
                        cv_log_placeholder = cv_log_expander.empty()
                        cv_st_handler = StreamlitLogHandler(cv_log_placeholder)
                        cv_st_handler.setLevel(logging.INFO)
                        cv_logger = logging.getLogger('cv_engine')
                        if not any(isinstance(h, StreamlitLogHandler) for h in cv_logger.handlers):
                            cv_logger.addHandler(cv_st_handler)
                        original_cv_level = cv_logger.level
                        cv_logger.setLevel(logging.INFO)
                    
                        best_model_cv = trainer.train(data_dir, n_epochs=epochs, lr=lr_cv, callback=cv_callback, mask_dir=mask_dir)
                        
                        cv_logger.removeHandler(cv_st_handler)
                        cv_logger.setLevel(original_cv_level)
                        
                        st.success("Vision Training Complete!")
                        st.session_state['best_cv_model'] = best_model_cv
                        st.session_state['cv_trainer'] = trainer
                        
                        # Store class names for inference display
                        if hasattr(trainer, 'class_names'):
                            st.session_state['cv_class_names'] = trainer.class_names
                        
                        # Log to MLflow
                        try:
                            import mlflow
                            import mlflow.pytorch
                            
                            # Use a clean run name without emojis to avoid Windows encoding issues
                            clean_run_name = f"CV_Task_{cv_task}"
                            
                            with mlflow.start_run(run_name=clean_run_name):
                                mlflow.log_param("task", cv_task)
                                mlflow.log_param("epochs", epochs)
                                mlflow.log_param("lr", lr_cv)
                                if hasattr(trainer, 'class_names'):
                                    mlflow.log_dict({"class_names": trainer.class_names}, "metadata/classes.json")
                                
                                # Log metrics from history
                                for entry in trainer.history:
                                    mlflow.log_metric("accuracy", entry['acc'], step=entry['epoch'])
                                    mlflow.log_metric("loss", entry['loss'], step=entry['epoch'])
                                
                                mlflow.pytorch.log_model(best_model_cv, "model")
                                st.info("Experiment logged to MLflow successfully.")
                        except Exception as ml_err:
                            st.warning(f"Failed to log to MLflow: {ml_err}")

                    except Exception as e:
                        try:
                            cv_logger.removeHandler(cv_st_handler)
                            cv_logger.setLevel(original_cv_level)
                        except:
                            pass
                        st.error(f"CV Training Failed: {e}")
                        st.error("Check if your ZIP file structure matches the task requirements (e.g., ImageFolder structure for classification).")

        if st.session_state.get('best_cv_model'):
            st.divider()
            st.subheader("Inference Test")
            test_img = st.file_uploader("Upload image for prediction", type=['jpg', 'png'], key="cv_test_upload")
            if test_img:
                img_path = f"temp_{test_img.name}"
                with open(img_path, "wb") as f:
                    f.write(test_img.getbuffer())
                
                trainer = st.session_state['cv_trainer']
                prediction = trainer.predict(img_path)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.image(test_img, caption="Uploaded Image")
                
                with col_res2:
                    if cv_task == "image_segmentation":
                        st.write("Segmentation Result:")
                        mask_img = Image.fromarray((prediction * (255 // (prediction.max() if prediction.max() > 0 else 1))).astype(np.uint8))
                        st.image(mask_img, caption="Predicted Mask", use_container_width=True)
                    elif cv_task == "object_detection":
                        st.write("Detection Result (Boxes):")
                        # Draw boxes on image
                        img_draw = Image.open(img_path).convert("RGB")
                        draw = ImageDraw.Draw(img_draw)
                        boxes = prediction['boxes'].cpu().numpy()
                        scores = prediction['scores'].cpu().numpy()
                        for box, score in zip(boxes, scores):
                            if score > 0.5: # Threshold
                                draw.rectangle(box, outline="red", width=3)
                                draw.text((box[0], box[1]), f"{score:.2f}", fill="red")
                        st.image(img_draw, caption="Predicted Objects", use_container_width=True)
                    else:
                        # Map ID to Class Name if available
                        class_names = st.session_state.get('cv_class_names')
                        if class_names and isinstance(prediction, int) and prediction < len(class_names):
                            st.metric("Predicted Class", class_names[prediction])
                        else:
                            st.metric("Predicted Class ID", prediction)
                
                try:
                    os.remove(img_path)
                except: pass

# --- TAB 3: EXPERIMENTS ---
with tabs[2]:
    jm: TrainingJobManager = st.session_state['job_manager']
    
    @st.fragment(run_every=2.0)
    def experiments_dashboard():
        jm.poll_updates()
        jobs = jm.list_jobs()

        # Header
        col_exp_h1, col_exp_h2 = st.columns([3, 1])
        with col_exp_h1:
            st.header("🧪 Experiments")
            st.caption(f"Active jobs: **{jm.active_count()}** · Total: **{len(jobs)}**")
        with col_exp_h2:
            if st.button("🔄 Refresh", key="exp_refresh"):
                st.rerun()

        if not jobs:
            st.info("No experiments yet. Go to **AutoML & Model Hub** → configure settings → click **Submit Experiment**.")
            return

        # Search / filter bar
        search_q = st.text_input("🔍 Search experiments", key="exp_search", placeholder="Filter by name...")
        filter_status = st.multiselect(
            "Status", 
            [JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
            default=[JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
            key="exp_status_filter",
            format_func=lambda s: {"queued":"🔵 Queued","running":"🟢 Running","paused":"🟡 Paused",
                                   "completed":"✅ Completed","failed":"🔴 Failed","cancelled":"⚫ Cancelled"}.get(s, s)
        )

        visible_jobs = [
            j for j in jobs
            if j.status in filter_status and (not search_q or search_q.lower() in j.name.lower())
        ]

        if not visible_jobs:
            st.warning("No experiments match the current filters.")
        
        for job in visible_jobs:
            # Status badge color
            badge = {"queued":"🔵","running":"🟢","paused":"🟡","completed":"✅","failed":"🔴","cancelled":"⚫"}.get(job.status, "⚪")
            score_str = f"Best: **{job.best_score:.4f}**" if job.best_score is not None else ""
            label = f"{badge} **{job.name}** &nbsp;&nbsp; `{job.status.upper()}` &nbsp;&nbsp; ⏱ {job.duration_str}  &nbsp;&nbsp; {score_str}"

            with st.expander(label, expanded=(job.status in (JobStatus.RUNNING, JobStatus.PAUSED))):
                # ──── Action buttons ────
                btn_cols = st.columns([1, 1, 1, 1, 1, 2])
                with btn_cols[0]:
                    if job.status == JobStatus.RUNNING:
                        if st.button("⏸ Pause", key=f"pause_{job.job_id}"):
                            jm.pause_job(job.job_id)
                            st.rerun()
                with btn_cols[1]:
                    if job.status == JobStatus.PAUSED:
                        if st.button("▶ Resume", key=f"resume_{job.job_id}"):
                            jm.resume_job(job.job_id)
                            st.rerun()
                with btn_cols[2]:
                    if job.is_active():
                        if st.button("⏹ Cancel", key=f"cancel_{job.job_id}"):
                            jm.cancel_job(job.job_id)
                            st.rerun()
                with btn_cols[3]:
                    if st.button("🗑 Delete", key=f"delete_{job.job_id}"):
                        jm.delete_job(job.job_id)
                        st.rerun()
                with btn_cols[4]:
                    if job.mlflow_run_id:
                        if st.button("📋 MLflow", key=f"mlflow_{job.job_id}"):
                            st.session_state[f'mlflow_sel_{job.job_id}'] = True

                # ──── Error display ────
                if job.status == JobStatus.FAILED and job.error_msg:
                    st.error(f"**Error:** {job.error_msg}")

                # ──── Detail sub-tabs ────
                detail_tabs = st.tabs(["📊 Overview", "📈 Progress", "🖥 Logs", "🏆 Results", "🔬 MLflow Details", "📦 Register"])

                # ── Tab 0: Overview ──
                with detail_tabs[0]:
                    ov1, ov2 = st.columns(2)
                    with ov1:
                        st.markdown(f"**Status:** {job.status_label}")
                        st.markdown(f"**Job ID:** `{job.job_id}`")
                        st.markdown(f"**Duration:** {job.duration_str}")
                        if job.best_score is not None:
                            st.metric("Best Score", f"{job.best_score:.4f}")
                        if job.mlflow_run_id:
                            st.markdown(f"**MLflow Run:** `{job.mlflow_run_id}`")
                        if job.mlflow_experiment:
                            st.markdown(f"**Experiment:** `{job.mlflow_experiment}`")
                    with ov2:
                        cfg = job.config
                        st.markdown("**Configuration:**")
                        st.json({
                            "task": cfg.get("task"),
                            "target": cfg.get("target"),
                            "preset": cfg.get("preset"),
                            "n_trials": cfg.get("n_trials"),
                            "validation": cfg.get("validation_strategy"),
                            "optimization_metric": cfg.get("optimization_metric"),
                            "selected_models": cfg.get("selected_models") or "All",
                        }, expanded=False)

                # ── Tab 1: Progress ──
                with detail_tabs[1]:
                    if job.trials_data:
                        df_t = pd.DataFrame(job.trials_data)
                        metric_col = job.target_metric
                        if metric_col not in df_t.columns:
                            numeric_cols = df_t.select_dtypes(include='number').columns.tolist()
                            metric_col = numeric_cols[-1] if numeric_cols else None
                        
                        if metric_col and "Model" in df_t.columns:
                            fig_p = px.line(
                                df_t, x="Model Trial", y=metric_col, color="Model",
                                markers=True, title=f"Optimization Progress — {metric_col}",
                                hover_name="Identifier" if "Identifier" in df_t.columns else None
                            )
                            st.plotly_chart(fig_p, use_container_width=True, key=f"prog_{job.job_id}")
                        
                        with st.expander("All Trials Table"):
                            st.dataframe(df_t.dropna(axis=1, how='all'), use_container_width=True)
                    else:
                        st.info("Waiting for first trial to complete…")
                    
                    # Model summaries if available
                    if job.model_summaries:
                        st.divider()
                        st.markdown("#### Best Results by Model")
                        sum_rows = []
                        for mn, mi in job.model_summaries.items():
                            row = {"Model": mn, "Best Score": f"{mi.get('score', 0):.4f}",
                                   "Trial": mi.get("trial_name", "?"), "Duration (s)": f"{mi.get('duration', 0):.2f}"}
                            if 'metrics' in mi:
                                for mk, mv in mi['metrics'].items():
                                    if mk != 'confusion_matrix' and isinstance(mv, (int, float)):
                                        row[mk.upper()] = f"{mv:.4f}"
                            sum_rows.append(row)
                        if sum_rows:
                            st.dataframe(pd.DataFrame(sum_rows), use_container_width=True)

                # ── Tab 2: Logs ──
                with detail_tabs[2]:
                    if job.logs:
                        log_text = "\n".join(job.logs[-200:])
                        st.text_area("Live Logs (last 200 lines)", value=log_text, height=400, key=f"logs_{job.job_id}", disabled=True)
                    else:
                        st.info("No logs yet. Logs appear as training progresses.")

                # ── Tab 3: Results ──
                with detail_tabs[3]:
                    if job.report_data:
                        for m_name, rep in job.report_data.items():
                            with st.expander(f"📊 Report: {m_name} (Score: {rep.get('score', 0):.4f})", expanded=True):
                                r1, r2 = st.columns([1, 2])
                                with r1:
                                    st.markdown("**Validation Metrics**")
                                    rep_metrics = rep.get('metrics', {})
                                    if rep_metrics:
                                        met_df = pd.DataFrame(list(rep_metrics.items()), columns=['Metric', 'Value'])
                                        met_df = met_df[met_df['Metric'] != 'confusion_matrix']
                                        st.table(met_df)
                                    st.markdown(f"**MLflow Run:** `{rep.get('run_id', 'N/A')}`")
                                    st.markdown(f"**Best Trial:** {rep.get('best_trial_number', 'N/A')}")

                                    # Stability results
                                    if rep.get('stability'):
                                        st.divider()
                                        st.markdown("**Stability Analysis**")
                                        for s_type, s_data in rep['stability'].items():
                                            with st.expander(f"Test: {s_type}"):
                                                if isinstance(s_data, dict):
                                                    st.json(s_data)
                                                else:
                                                    st.dataframe(s_data)

                                with r2:
                                    plots = rep.get('plots', {})
                                    if plots:
                                        plot_names = list(plots.keys())
                                        plot_tabs = st.tabs(plot_names)
                                        for pi, pn in enumerate(plot_names):
                                            with plot_tabs[pi]:
                                                ptype, pbuf = plots[pn]
                                                try:
                                                    import io as _io
                                                    st.image(_io.BytesIO(pbuf))
                                                except Exception:
                                                    st.warning(f"Could not render plot: {pn}")
                                    else:
                                        st.info("No plots available.")

                    elif job.status == JobStatus.COMPLETED:
                        cfg = job.config
                        bp = cfg.get('best_params', {})
                        ev = cfg.get('eval_metrics', {})
                        st.markdown(f"**Best Model:** `{bp.get('model_name', 'Unknown')}`")
                        if ev:
                            st.markdown("**Evaluation Metrics:**")
                            for k, v in ev.items():
                                if k != 'confusion_matrix' and isinstance(v, (int, float)):
                                    st.metric(k.upper(), f"{v:.4f}")
                        cc = cfg.get('consumption_code')
                        if cc:
                            st.divider()
                            st.markdown("**Model Consumption Code:**")
                            st.code(cc, language='python')
                    else:
                        st.info("Results will appear here when the experiment completes.")

                # ── Tab 4: MLflow Details ──
                with detail_tabs[4]:
                    if job.mlflow_run_id:
                        # Caching MLflow data
                        cache = st.session_state['mlflow_cache']
                        if job.mlflow_run_id not in cache:
                            with st.spinner("Fetching MLflow data…"):
                                cache[job.mlflow_run_id] = get_run_details(job.mlflow_run_id)
                        
                        rd = cache[job.mlflow_run_id]
                        
                        if "error" in rd:
                            st.error(f"Could not load MLflow data: {rd['error']}")
                        else:
                            info = rd.get("info", {})
                            cols_ml = st.columns(3)
                            cols_ml[0].metric("Status", info.get("status", "?"))
                            if info.get("start_time"):
                                import datetime as _dt
                                st.datetime_info = _dt.datetime.fromtimestamp(info["start_time"] / 1000)
                                cols_ml[1].metric("Started", st.datetime_info.strftime("%H:%M:%S"))
                            cols_ml[2].caption(f"Experiment: **{info.get('experiment_name', '?')}**")

                            ml_tabs = st.tabs(["Parameters", "Metrics", "Tags", "Artifacts"])
                            with ml_tabs[0]:
                                params_data = rd.get("params", {})
                                if params_data:
                                    params_df = pd.DataFrame(list(params_data.items()), columns=['Parameter', 'Value'])
                                    st.dataframe(params_df, use_container_width=True)
                                else:
                                    st.info("No parameters logged.")
                            with ml_tabs[1]:
                                metrics_data = rd.get("metrics", {})
                                if metrics_data:
                                    met_cols = st.columns(min(len(metrics_data), 4))
                                    for mi, (mk, mv) in enumerate(metrics_data.items()):
                                        met_cols[mi % 4].metric(mk, f"{mv:.4f}" if isinstance(mv, float) else mv)
                                    
                                    # Show history chart for numeric metrics
                                    hist = rd.get("metric_history", {})
                                    if hist:
                                        metric_sel = st.selectbox("Metric History", list(hist.keys()), key=f"mhist_{job.job_id}")
                                        if metric_sel and hist.get(metric_sel):
                                            hist_df = pd.DataFrame(hist[metric_sel])
                                            fig_hist = px.line(hist_df, x="step", y="value", title=f"{metric_sel} History")
                                            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{job.job_id}")
                                else:
                                    st.info("No metrics logged.")
                            with ml_tabs[2]:
                                tags_data = rd.get("tags", {})
                                if tags_data:
                                    st.json(tags_data)
                                else:
                                    st.info("No custom tags.")
                            with ml_tabs[3]:
                                artifacts = rd.get("artifacts", [])
                                if artifacts:
                                    for a in artifacts:
                                        st.markdown(f"📄 `{a}`")
                                else:
                                    st.info("No artifacts logged.")
                    else:
                        st.info("MLflow data available after the experiment completes.")

                # ── Tab 5: Register Model ──
                with detail_tabs[5]:
                    if job.mlflow_run_id:
                        st.markdown("### Register this run as an Official Model")
                        reg_name = st.text_input(
                            "Model Registry Name", 
                            value=f"{job.config.get('task', 'model')}_{job.name[:20]}",
                            key=f"reg_name_{job.job_id}"
                        )
                        if st.button("📦 Register Model", key=f"reg_btn_{job.job_id}", type="primary"):
                            with st.spinner("Registering…"):
                                success = register_model_from_run(job.mlflow_run_id, reg_name)
                            if success:
                                st.success(f"✅ Model **{reg_name}** registered! Check the **Model Registry & Deploy** tab.")
                            else:
                                st.error("Registration failed. Ensure the run has a logged model artifact.")
                    else:
                        st.info("Model registration is available after the experiment completes and logs a model to MLflow.")

    # Call the dashboard fragment
    experiments_dashboard()

    # ── Historic MLflow runs not in job manager ──
    st.divider()
    with st.expander("📚 All MLflow Runs (Historic)", expanded=False):
        try:
            runs = get_cached_all_runs()
            if not runs.empty:
                exp_names = runs['experiment_name'].unique().tolist()
                sel_exps = st.multiselect("Filter Experiments", exp_names, default=exp_names, key="hist_exp_filter")
                filt_runs = runs[runs['experiment_name'].isin(sel_exps)].sort_values('start_time', ascending=False)
                st.dataframe(filt_runs[['run_id', 'experiment_name', 'status', 'start_time']], use_container_width=True)
                
                run_id_sel = st.selectbox("Select Run to Register", filt_runs['run_id'].tolist(), key="hist_run_sel")
                if run_id_sel:
                    reg_name_hist = st.text_input("Registry Name", value=f"model_{run_id_sel[:6]}", key="hist_reg_name")
                    if st.button("Register Selected Run", key="hist_reg_btn"):
                        if register_model_from_run(run_id_sel, reg_name_hist):
                            st.success(f"Model {reg_name_hist} registered!")
                            st.rerun()
            else:
                st.info("No historic MLflow runs found.")
        except Exception as e:
            st.warning(f"Could not load historic runs: {e}")

# --- TAB 3: MODEL REGISTRY & DEPLOY ---
with tabs[3]:
    st.header("Model Registry & Deployment")
    
    st.subheader("Model Deployment Center")
    
# 1. Select Model from Registry
    models = get_registered_models()
    if not models:
        st.warning("No models found in Registry. Please register a model from Experiments first.")
    else:
        col_dep1, col_dep2 = st.columns([1, 2])
        
        with col_dep1:
            st.markdown("#### 1. Select Artifact")
            model_names = [m.name for m in models]
            selected_model_name = st.selectbox("Registered Model", model_names, key="deploy_model_sel")
            
            # Get versions
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{selected_model_name}'")
            version_nums = [v.version for v in versions]
            selected_version = st.selectbox("Version", version_nums, key="deploy_ver_sel")
            
            # Fetch details
            model_details = get_model_details(selected_model_name, selected_version)
            # model_details is a dictionary, not an object
            st.info(f"Stage: {model_details.get('current_stage', 'None')}")
            st.caption(f"Last Updated: {model_details.get('last_updated_timestamp', 'Unknown')}")

        with col_dep2:
            st.markdown("#### 2. Deployment Configuration")
            env = st.radio("Target Environment", ["Development (Local)", "Staging", "Production"], horizontal=True)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                cpu_alloc = st.slider("CPU Allocation", 0.5, 4.0, 1.0, step=0.5)
                min_replicas = st.number_input("Min Replicas", 1, 5, 1)
            with col_res2:
                mem_alloc = st.slider("Memory (GB)", 0.5, 16.0, 2.0, step=0.5)
                max_replicas = st.number_input("Max Replicas", 1, 10, 2)
            
            if st.button("Deploy / Update Service", type="primary"):
                with st.spinner(f"Deploying {selected_model_name} v{selected_version} to {env}..."):
                    # Mock Deployment Process
                    time.sleep(2)
                    endpoint_url = f"http://localhost:8000/predict/{selected_model_name}/{selected_version}"
                    st.session_state['active_endpoint'] = {
                        'url': endpoint_url,
                        'model': selected_model_name,
                        'version': selected_version,
                        'env': env,
                        'status': 'Healthy'
                    }
                    st.success(f"Deployment Successful! Endpoint active at: {endpoint_url}")
        
        st.divider()
        
        # 3. Integrated Testing Interface (Merged from old Test Models tab)
        st.markdown("#### 3. Live Inference Test")
        
        if 'active_endpoint' in st.session_state:
            st.success(f"Connected to Active Endpoint: `{st.session_state['active_endpoint']['url']}`")
            
            # Input Method
            input_type = st.radio("Input Format", ["JSON/Text", "CSV File"], horizontal=True)
            
            if input_type == "JSON/Text":
                input_data = st.text_area("Input JSON", value='{"feature1": 0.5, "feature2": 1.2}', height=150)
                if st.button("Send Request"):
                    # Mock Inference
                    try:
                        import json
                        data = json.loads(input_data)
                        # In real scenario: requests.post(endpoint, json=data)
                        # Here we load model locally to simulate
                        with st.spinner("Processing..."):
                            loaded_model = load_registered_model(selected_model_name, selected_version)
                            # Convert dict to DF
                            df_in = pd.DataFrame([data])
                            pred = loaded_model.predict(df_in)
                            st.json({"prediction": pred.tolist(), "latency_ms": 45, "model_version": selected_version})
                    except Exception as e:
                        st.error(f"Inference Error: {e}")
                        
            elif input_type == "CSV File":
                up_file = st.file_uploader("Upload Batch CSV", type="csv", key="deploy_test_csv")
                if up_file:
                    df_test = pd.read_csv(up_file)
                    if st.button("Run Batch Prediction"):
                        with st.spinner("Running batch inference..."):
                            loaded_model = load_registered_model(selected_model_name, selected_version)
                            preds = loaded_model.predict(df_test)
                            df_test['prediction'] = preds
                            st.dataframe(df_test)
                            st.info(f"Processed {len(df_test)} records in 0.42s")
        else:
                st.info("Deploy a model above to enable live testing.")





