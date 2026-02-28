from automl_engine import AutoMLDataProcessor, AutoMLTrainer
# from cv_engine import CVAutoMLTrainer, get_cv_explanation # Moved to local scope
import streamlit as st
import pandas as pd
import numpy as np
from mlops_utils import (
    MLFlowTracker, DriftDetector, ModelExplainer, get_model_registry, 
    DataLake, register_model_from_run, get_registered_models, get_all_runs,
    get_model_details, load_registered_model
)
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

class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler that streams logs directly to a Streamlit placeholder.
    """
    def __init__(self, placeholder, max_lines=50):
        super().__init__()
        self.placeholder = placeholder
        self.max_lines = max_lines
        self.log_lines = []
        # Format the log message slightly richer
        self.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_lines.append(msg)
            if len(self.log_lines) > self.max_lines:
                self.log_lines.pop(0)
            
            # Combine lines and write to placeholder
            log_text = '\n'.join(self.log_lines)
            # Use atomic update
            self.placeholder.code(log_text, language='text')
        except Exception:
            self.handleError(record)


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
    "Model Registry & Deploy"
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
                if st.button("Load into Workspace"):
                    st.session_state['df'] = datalake.load_version(selected_ds, selected_ver)
                    st.success(f"Loaded {selected_ds} ({selected_ver})")
                    st.rerun()

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

            if st.button("🚀 Start Training", key="btn_start_train"):
                st.session_state['trials_data'] = []
                st.session_state['report_data'] = {} # Reset reports on new training
                start_time_train = time.time()
                
                # Nome do experimento baseado no dataset e timestamp
                exp_tag = selected_configs[0]['name'] if selected_configs else "AutoML"
                experiment_name = f"{exp_tag}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"

                # Container for feedback
                status_c = st.empty()
                progress_bar = st.progress(0)
                chart_c = st.empty()
                
                # Container for real-time logs
                st.markdown("### 🖥️ Execution Logs (Real-time)")
                log_expander = st.expander("View AutoML Engine and Optuna Logs", expanded=True)
                log_placeholder = log_expander.empty()
                
                # Setup Streamlit Logging Handler
                st_handler = StreamlitLogHandler(log_placeholder)
                st_handler.setLevel(logging.INFO)
                
                # Attach to both automl_engine and optuna loggers
                automl_logger = logging.getLogger('automl_engine')
                optuna_logger = logging.getLogger('optuna')
                
                # Prevent adding multiple handlers if button clicked multiple times
                if not any(isinstance(h, StreamlitLogHandler) for h in automl_logger.handlers):
                    automl_logger.addHandler(st_handler)
                if not any(isinstance(h, StreamlitLogHandler) for h in optuna_logger.handlers):
                    optuna_logger.addHandler(st_handler)
                
                # Temporarily set log level to INFO to capture the progress
                original_automl_level = automl_logger.level
                original_optuna_level = optuna_logger.level
                automl_logger.setLevel(logging.INFO)
                optuna_logger.setLevel(logging.INFO)
                
                # Container for per-model reports (NEW)
                st.markdown("### 📊 Per-Model Reports (Real-time)")
                if 'report_data' not in st.session_state:
                    st.session_state['report_data'] = {}
                
                # Use st.empty() as a fixed placeholder for the entire report section
                report_section_placeholder = st.empty()

                # Calcular total real de trials para a barra de progresso
                # Instancia o trainer com o preset para pegar as configs
                trainer_for_info = AutoMLTrainer(task_type=task, preset=training_preset)
                preset_config = trainer_for_info.preset_configs.get(training_preset)
                
                effective_models_list = selected_models if selected_models else preset_config['models']
                n_trials_val = n_trials if n_trials is not None else preset_config['n_trials']
                total_expected_trials = n_trials_val * len(effective_models_list)
                
                def callback(trial, score, full_name, dur, metrics=None):
                    # Check for special report event
                    if metrics and '__report__' in metrics:
                        report = metrics['__report__']
                        model_name = report['model_name']
                        
                        # Store report in session state to survive reruns
                        st.session_state['report_data'][model_name] = report
                        
                        # Atomic render: replace the entire content of the placeholder
                        with report_section_placeholder.container():
                            # Render ALL reports stored in session state
                            for m_name, rep in st.session_state['report_data'].items():
                                expander_label = f"Final Report: {rep['model_name']} (Score: {rep['score']:.4f})"
                                with st.expander(expander_label, expanded=True):
                                    col_rep1, col_rep2 = st.columns([1, 2])
                                    with col_rep1:
                                        st.markdown("🎯 **Validation Metrics**")
                                        if rep['metrics']:
                                            # Display metrics as a nice table instead of raw JSON
                                            met_df = pd.DataFrame([rep['metrics']]).T.reset_index()
                                            met_df.columns = ['Metric', 'Value']
                                            st.table(met_df)
                                        else:
                                            st.warning("Validation metrics not available.")
                                            
                                        st.markdown(f"📌 **Best Trial:** {rep['best_trial_number']}")
                                        st.markdown(f"🔗 **MLflow Run ID:** `{rep['run_id']}`")
                                        
                                        if 'stability' in rep and rep['stability']:
                                            st.divider()
                                            st.markdown("⚖️ **Stability Analysis**")
                                            for s_type, s_data in rep['stability'].items():
                                                if s_type == 'general':
                                                    st.write("📈 **General Stability Summary**")
                                                    summary = {k: v for k, v in s_data.items() if k not in ['raw_seed', 'raw_split']}
                                                    # Convert nested dict to flat for table
                                                    flat_summary = []
                                                    for metric, stats in summary.items():
                                                        if isinstance(stats, dict):
                                                            row = {'Metric': metric.upper()}
                                                            row.update({k.capitalize(): f"{v:.4f}" if isinstance(v, float) else v for k, v in stats.items()})
                                                            flat_summary.append(row)
                                                    
                                                    if flat_summary:
                                                        st.table(pd.DataFrame(flat_summary))
                                                elif s_type == 'hyperparam':
                                                    with st.expander(f"Sensitivity: {s_type}", expanded=False):
                                                        st.dataframe(s_data, use_container_width=True)
                                                else:
                                                    with st.expander(f"Test: {s_type}", expanded=False):
                                                        st.dataframe(s_data, use_container_width=True)
                                    with col_rep2:
                                        if 'plots' in rep and rep['plots']:
                                            plot_labels = list(rep['plots'].keys())
                                            if plot_labels:
                                                tab_plots = st.tabs(plot_labels)
                                                for i, plot_name in enumerate(plot_labels):
                                                    with tab_plots[i]:
                                                        # Handle both Matplotlib Figures and PIL Images
                                                        plot_obj = rep['plots'][plot_name]
                                                        if isinstance(plot_obj, Image.Image):
                                                            st.image(plot_obj, use_container_width=True)
                                                        else:
                                                            st.pyplot(plot_obj, clear_figure=True)
                                            else:
                                                st.info("No visualizations available for this model.")
                                        else:
                                            st.info("No visualizations available for this model.")
                        return

                    # Extrair nome do algoritmo e o número do trial do modelo
                    algo_name = full_name.split(" - ")[0]
                    trial_label = full_name.split(" - ")[1] # "Trial X"
                    trial_num = int(trial_label.replace("Trial ", ""))

                    trial_info = {
                        "Global Trial": trial.number + 1,
                        "Model Trial": trial_num,
                        "Model": algo_name,
                        "Identifier": full_name,
                        "Duration (s)": dur
                    }

                    # Adicionar métricas adicionais se houver
                    if metrics:
                        for m_k, m_v in metrics.items():
                            if m_k != "__report__" and isinstance(m_v, (int, float, np.number)):
                                col_name = m_k.upper()
                                trial_info[col_name] = m_v
                    
                    # Garantir que a métrica alvo esteja presente no trial_info para o gráfico
                    if target_metric_name not in trial_info:
                        trial_info[target_metric_name] = score

                    st.session_state['trials_data'].append(trial_info)
                    
                    df_trials = pd.DataFrame(st.session_state['trials_data'])
                    
                    with status_c:
                        metric_text = f"{target_metric_name}: {score:.4f}"
                        st.info(f"{full_name} completed | {metric_text} | Total: {trial.number + 1}/{total_expected_trials}")
                    
                    progress_bar.progress(min((trial.number + 1) / total_expected_trials, 1.0))
                    
                    with chart_c:
                        # Gráfico mostrando o progresso de cada modelo individualmente
                        # Prepare rich hover data
                        available_cols = df_trials.columns.tolist()
                        hover_data_cols = [c for c in ["Model", "Identifier", "Duration (s)"] if c in available_cols]
                        
                        # Adicionar todas as métricas encontradas ao hover
                        for col in available_cols:
                            if col not in hover_data_cols and col not in ["Global Trial", "Model Trial"]:
                                hover_data_cols.append(col)
                        
                        fig = px.line(df_trials, x="Model Trial", y=target_metric_name, color="Model", 
                                    markers=True, 
                                    hover_name="Identifier",
                                    hover_data=hover_data_cols,
                                    title=f"Optimization Progress: {target_metric_name} by Algorithm")
                        
                        fig.update_layout(xaxis_title="Model Trial No.", yaxis_title=target_metric_name)
                        st.plotly_chart(fig, key=f"chart_{trial.number}", use_container_width=True)

                with st.spinner("Processing..."):
                    processor = AutoMLDataProcessor(target_column=target, task_type=task, date_col=date_col_pre, forecast_horizon=forecast_horizon, nlp_config=nlp_config_automl)
                    X_train_proc, y_train_proc = processor.fit_transform(train_df, nlp_cols=selected_nlp_cols)
                    
                    # Exibir relatório de qualidade Deepchecks se disponível
                    if hasattr(processor, 'quality_report_html') and processor.quality_report_html:
                        with st.expander("📊 Data Quality Report (Deepchecks)", expanded=False):
                            import streamlit.components.v1 as components
                            # Limitar altura para não quebrar a UI
                            components.html(processor.quality_report_html, height=800, scrolling=True)
                    
                    X_test_proc, y_test_proc = processor.transform(test_df) if test_df is not None else (None, None)
                    
                    # Preparar modelos customizados (Upload/Registry)
                    custom_models = {}
                    if model_source == "Local Upload (.pkl)" and 'uploaded_pkl' in locals() and uploaded_pkl:
                         try:
                             loaded_model = joblib.load(uploaded_pkl)
                             custom_models["Uploaded_Model"] = loaded_model
                         except Exception as e:
                             st.error(f"Error loading .pkl: {e}")
                             st.stop()
                    elif model_source == "Model Registry (Registered)" and selected_models:
                         model_name = selected_models[0]
                         try:
                             loaded_model = load_registered_model(model_name)
                             custom_models[model_name] = loaded_model
                         except Exception as e:
                             st.error(f"Error loading from registry: {e}")
                             st.stop()

                    trainer = AutoMLTrainer(task_type=task, preset=training_preset, ensemble_config=ensemble_config)
                    
                    # Limpar nome do experimento para evitar erro de codificação no Windows
                    clean_experiment_name = "".join(c for c in experiment_name if ord(c) < 128)
                    if not clean_experiment_name:
                        clean_experiment_name = "AutoML_Experiment"

                    # Get feature and class names for better reporting
                    feature_names = processor.get_feature_names()
                    class_names = None
                    if hasattr(processor, 'label_encoder') and processor.label_encoder:
                        class_names = processor.label_encoder.classes_.tolist()

                    try:
                        best_model = trainer.train(
                            X_train_proc, 
                            y_train_proc, 
                            n_trials=n_trials,
                            timeout=timeout_per_model,
                            time_budget=total_time_budget,
                            callback=callback, 
                            selected_models=selected_models, 
                            early_stopping_rounds=early_stopping,
                            manual_params=manual_params,
                            experiment_name=clean_experiment_name,
                            random_state=random_seed_config,
                            validation_strategy=validation_strategy,
                            validation_params=validation_params,
                            custom_models=custom_models,
                            optimization_mode=selected_opt_mode,
                            optimization_metric=optimization_metric,
                            stability_config={'tests': selected_stability_tests, 'n_iterations': 3} if enable_stability else None,
                            feature_names=feature_names,
                            class_names=class_names
                        )
                        best_params = trainer.best_params
                        
                        st.session_state['best_model'] = best_model
                        st.session_state['best_params'] = best_params
                        st.session_state['processor'] = processor
                        
                        # Evaluation
                        metrics, y_pred = trainer.evaluate(X_test_proc, y_test_proc) if X_test_proc is not None else (None, None)
                        
                        st.success("AutoML Process Finished Successfully!")
                    finally:
                        # Ensure the Streamlit handler is removed after process finishes or fails
                        try:
                            automl_logger.removeHandler(st_handler)
                            optuna_logger.removeHandler(st_handler)
                            automl_logger.setLevel(original_automl_level)
                            optuna_logger.setLevel(original_optuna_level)
                        except:
                            pass
                    
                    # Mostrar o melhor modelo de forma destacada
                    best_model_name = trainer.best_params.get('model_name', 'Unknown')
                    st.balloons()
                    st.markdown(f"""
                        <div style="background-color:#d4edda; padding:20px; border-radius:10px; border-left:8px solid #28a745; margin-bottom:20px;">
                            <h2 style="color:#155724; margin:0;">Best Model Found: {best_model_name}</h2>
                            <p style="color:#155724; font-size:1.1em; margin-top:10px;">The system optimized and selected the algorithm above as the best performing for your task.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # --- Resumo por Modelo ---
                    if hasattr(trainer, 'model_summaries') and trainer.model_summaries:
                        st.markdown("### Best Results by Algorithm")
                        summary_data = []
                        for m_name, info in trainer.model_summaries.items():
                            row = {
                                "Algorithm": m_name,
                                "Best Score": f"{info['score']:.4f}",
                                "Trial": info['trial_name'],
                                "Duration (s)": f"{info['duration']:.2f}"
                            }
                            # Adicionar métricas adicionais se disponíveis
                            if 'metrics' in info:
                                for met_name, met_val in info['metrics'].items():
                                    if met_name != 'confusion_matrix' and isinstance(met_val, (int, float, np.number)):
                                        row[met_name.upper()] = f"{met_val:.4f}"
                            summary_data.append(row)
                        
                        df_summary = pd.DataFrame(summary_data)
                        st.table(df_summary)
                        
                        # Também permitir ver todos os trials em uma tabela expansível
                        with st.expander("View Full History of All Trials"):
                            if st.session_state.get('trials_data'):
                                df_all = pd.DataFrame(st.session_state['trials_data'])
                                if not df_all.empty:
                                    # Drop all-NA columns to avoid deprecation warning
                                    df_all = df_all.dropna(axis=1, how='all')
                                    st.dataframe(df_all.sort_values(by=target_metric_name, ascending=False), use_container_width=True)
                            else:
                                st.info("No trial data to display.")

                    if metrics: 
                        st.markdown("### Final Results (Global Best Model)")
                        cols_m = st.columns(len(metrics))
                        for i, (m_name, m_val) in enumerate(metrics.items()):
                            if m_name != 'confusion_matrix':
                                with cols_m[i % len(cols_m)]:
                                    st.metric(m_name.upper(), f"{m_val:.4f}" if isinstance(m_val, (float, np.float64, np.float32)) else m_val)
                    
                    # --- Visualizações de Resultados ---
                    if X_test_proc is not None:
                        st.divider()
                        st.subheader("Performance Visualization")
                        
                        if task == "classification":
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                if 'confusion_matrix' in metrics:
                                    cm = np.array(metrics['confusion_matrix'])
                                    
                                    # Use class names for Plotly labels if available
                                    labels_plotly = class_names if class_names else None
                                    
                                    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                                                     labels=dict(x="Predicted", y="Actual", color="Quantity"),
                                                     x=labels_plotly, y=labels_plotly)
                                    st.plotly_chart(fig_cm)
                            with col_v2:
                                # Feature Importance (SHAP - SHapley Additive exPlanations)
                                st.markdown("#### Feature Importance (SHAP)")
                                st.info("Calculating Explainability via SHAP (may take a few seconds)...")
                                
                                shap_success = False
                                try:
                                    # Usar sample para performance em ambientes com poucos recursos
                                    max_shap_train = 100
                                    max_shap_test = 50
                                    
                                    sample_train = X_train_proc
                                    if len(sample_train) > max_shap_train:
                                        sample_train = shap.utils.sample(sample_train, max_shap_train)
                                        
                                    sample_test = X_test_proc
                                    if sample_test is not None and len(sample_test) > max_shap_test:
                                        sample_test = shap.utils.sample(sample_test, max_shap_test)

                                    if sample_test is not None:
                                        # Use dense representation if sparse to avoid SHAP errors
                                        if hasattr(sample_train, "toarray"):
                                            sample_train = sample_train.toarray()
                                        if hasattr(sample_test, "toarray"):
                                            sample_test = sample_test.toarray()
                                            
                                        explainer = ModelExplainer(best_model, sample_train, task_type=task)
                                        
                                        # Plot Beeswarm (Resumo)
                                        st.markdown("**SHAP Summary Plot**")
                                        st.caption("Mostra como cada feature impacta a saida do modelo. Pontos vermelhos = valor alto da feature, azuis = valor baixo.")
                                        fig_shap = explainer.plot_importance(sample_test, plot_type="summary")
                                        if fig_shap:
                                            st.pyplot(fig_shap, clear_figure=True)
                                        
                                        # Plot Bar (Global Importance)
                                        st.markdown("**SHAP Feature Importance (Bar)**")
                                        st.caption("Absolute mean of the impact of each feature.")
                                        fig_shap_bar = explainer.plot_importance(sample_test, plot_type="bar")
                                        if fig_shap_bar:
                                            st.pyplot(fig_shap_bar, clear_figure=True)
                                        shap_success = True
                                except Exception as e:
                                    st.warning(f"Could not generate SHAP plot: {e}")

                                # Fallback to manual feature importance if SHAP fails
                                if not shap_success and hasattr(trainer, 'feature_importance') and trainer.feature_importance:
                                    st.info("Displaying importance based on coefficients/trees (alternative method).")
                                    fi_data = pd.DataFrame({
                                        'Feature': processor.get_feature_names(),
                                        'Importance': trainer.feature_importance
                                    }).sort_values(by='Importance', ascending=False)
                                    
                                    fig_fi = px.bar(fi_data.head(15), x='Importance', y='Feature', orientation='h',
                                                  title="Top 15 Most Important Features")
                                    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig_fi, use_container_width=True)

                        elif task in ["regression", "time_series"]:
                            df_res = pd.DataFrame({"Actual": y_test_proc, "Predicted": y_pred})
                            if task == "time_series":
                                fig_res = px.line(df_res.reset_index(), y=["Actual", "Predicted"], title="Time Series: Actual vs Predicted")
                            else:
                                fig_res = px.scatter(df_res, x="Actual", y="Predicted", trendline="ols", title="Regression: Actual vs Predicted")
                            st.plotly_chart(fig_res)

                        elif task == "clustering":
                            # PCA for visualization
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_test_proc)
                            df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                            df_pca['Cluster'] = y_pred.astype(str)
                            fig_cluster = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title="Cluster Visualization (PCA)")
                            st.plotly_chart(fig_cluster)

                        elif task == "anomaly_detection":
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_test_proc)
                            df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                            # y_pred: -1 for anomaly, 1 for normal
                            df_pca['Status'] = np.where(y_pred == -1, 'Anomaly', 'Normal')
                            fig_anom = px.scatter(df_pca, x='PCA1', y='PCA2', color='Status', 
                                                color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
                                                title="Anomaly Detection (PCA)")
                            st.plotly_chart(fig_anom)



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
    st.header("Experiments Explorer")
    st.markdown("Here you can find the history of **all training runs**. Choose the best ones to register in the official catalog.")
    
    runs = get_cached_all_runs()
    if not runs.empty:
        # Filtros de Experimento
        exp_names = runs['experiment_name'].unique().tolist()
        selected_exps = st.multiselect("Filter Experiments", exp_names, default=exp_names)
        
        filtered_runs = runs[runs['experiment_name'].isin(selected_exps)].sort_values('start_time', ascending=False)
        
        # Grid de Runs
        st.dataframe(filtered_runs[['run_id', 'experiment_name', 'status', 'start_time']], use_container_width=True)
        
        st.divider()
        
        # Detalhes e Registro
        col_det1, col_det2 = st.columns([1, 1])
        with col_det1:
            run_id_sel = st.selectbox("Select Run to Explore", filtered_runs['run_id'].tolist())
            if run_id_sel:
                run_data = filtered_runs[filtered_runs['run_id'] == run_id_sel].iloc[0]
                st.markdown("#### Metrics")
                metrics = {k.replace('metrics.', ''): v for k, v in run_data.items() if k.startswith('metrics.') and pd.notna(v)}
                st.json(metrics)
        
        with col_det2:
            if run_id_sel:
                st.markdown("#### Register as Official Model")
                model_reg_name = st.text_input("Registry Name", value=f"model_{run_id_sel[:6]}")
                if st.button("Confirm Registration"):
                    if register_model_from_run(run_id_sel, model_reg_name):
                        st.success(f"Model {model_reg_name} is now in the Registry!")
                        st.rerun()
    else:
        st.info("No experiments found. Start a training session in the AutoML & Model Hub tab.")

# --- TAB 3: MODEL REGISTRY & DEPLOY ---
with tabs[3]:
    st.header("Model Registry, Deployment & Monitoring")
    
    reg_tabs = st.tabs(["Registry & Deploy", "Real-time Monitoring"])
    
    with reg_tabs[0]:
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

    with reg_tabs[1]:
        st.subheader("Model Performance Monitoring")
        
        if 'active_endpoint' in st.session_state:
            ep = st.session_state['active_endpoint']
            st.markdown(f"**Monitored Service:** {ep['model']} (v{ep['version']}) | **Env:** {ep['env']}")
            
            # Dashboard Mockup
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Latency", "42ms", "-5ms")
            m2.metric("Requests/min", "128", "+12")
            m3.metric("Error Rate", "0.02%", "0.00%")
            m4.metric("CPU Usage", "45%", "+2%")
            
            st.markdown("#### Prediction Drift & Accuracy")
            # Mock Charts
            chart_data = pd.DataFrame(np.random.randn(50, 2), columns=['latency', 'accuracy'])
            st.line_chart(chart_data)
            
            st.markdown("#### Alerts & Logs")
            st.error("2023-10-27 14:30: High Latency Warning (>200ms) detected on pod-2")
            st.info("2023-10-27 14:00: Autoscaling triggered (replicas 2 -> 3)")
            
        else:
            st.warning("No active deployment found. Deploy a model to view monitoring dashboard.")





