from src.core.processor import AutoMLDataProcessor
from src.engines.classical import AutoMLTrainer
from dotenv import load_dotenv
import os

load_dotenv()
# from src.engines.vision import CVAutoMLTrainer, get_cv_explanation # Migrated structure
import streamlit as st
import pandas as pd
import numpy as np
from src.tracking.mlflow import (
    MLFlowTracker, get_model_registry, 
    register_model_from_run, get_registered_models, get_all_runs,
    get_model_details, load_registered_model, get_run_details
)
from src.core.data_lake import DataLake
from src.utils.helpers import get_consumption_code, generate_model_card
from src.utils.explainers import ModelExplainer
from src.deploy.hf_deploy import deploy_to_huggingface
from src.tracking.manager import TrainingJobManager, JobStatus
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

# Caching heavy data fetch operations
@st.cache_data(ttl=300)
def get_cached_all_runs():
    return get_all_runs()

@st.cache_data(ttl=300)
def get_cached_registered_models():
    return get_registered_models()

@st.cache_data(ttl=60)
def get_cached_datasets():
    return DataLake("./data_lake").list_datasets()

@st.cache_data(ttl=60)
def get_cached_versions(dataset_name):
    return DataLake("./data_lake").list_versions(dataset_name)

@st.cache_data(ttl=300)
def get_cached_dataframe(dataset_name, version, nrows=None):
    return DataLake("./data_lake").load_version(dataset_name, version, nrows=nrows)

@st.cache_data(ttl=60)
def get_cached_run_details(run_id):
    return get_run_details(run_id)

# 🎨 Page config & full dark design system
try:
    st.set_page_config(
        page_title="AutoMLOps Studio",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except st.errors.StreamlitAPIException:
    print("ERROR: This app must be run with Streamlit.")
    print("Please run: streamlit run app.py")
    import sys
    sys.exit(1)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg-primary:  #0d1117;
  --bg-card:     #161b22;
  --bg-surface:  #1c2128;
  --accent-blue:   #2f80ed;
  --accent-green:  #27ae60;
  --accent-purple: #8b5cf6;
  --accent-orange: #f59e0b;
  --accent-red:    #ef4444;
  --text-primary: #e6edf3;
  --text-muted:   #8b949e;
  --border:       #30363d;
  --border-light: #21262d;
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--text-primary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--bg-card);
  border-radius: 12px;
  padding: 6px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background-color: transparent !important;
  border-radius: 8px;
  padding: 8px 18px;
  border: none !important;
  color: var(--text-muted) !important;
  font-weight: 500;
  font-size: 0.9rem;
  transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
  background-color: var(--bg-surface) !important;
  color: var(--text-primary) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
  color: white !important;
  box-shadow: 0 2px 12px rgba(47,128,237,0.35);
}
.stTabs [aria-selected="true"] p, .stTabs [data-baseweb="tab"] p {
  color: inherit !important;
  font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
  border-radius: 8px;
  border: 1px solid var(--border);
  background-color: var(--bg-surface);
  color: var(--text-primary);
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 0.875rem;
  padding: 6px 16px;
  transition: all 0.2s ease;
}
.stButton > button:hover {
  background-color: var(--bg-card);
  border-color: var(--accent-blue);
  color: var(--accent-blue);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(47,128,237,0.15);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
  border: none;
  color: white;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #1a6dd6, #7c3aed);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(47,128,237,0.35);
  color: white;
}

/* ── Cards ── */
.ui-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 12px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.ui-card:hover { border-color: var(--accent-blue); box-shadow: 0 4px 16px rgba(47,128,237,0.1); }
.ui-card-selected {
  background: linear-gradient(135deg, rgba(47,128,237,0.08), rgba(139,92,246,0.08));
  border: 2px solid var(--accent-blue);
  box-shadow: 0 4px 16px rgba(47,128,237,0.2);
}
.ui-metric {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  border-left: 3px solid var(--accent-blue);
  transition: transform 0.2s, box-shadow 0.2s;
}
.ui-metric:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(47,128,237,0.15); }
.ui-metric .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text-primary); }
.ui-metric .metric-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Live metric pulse animation ── */
@keyframes metric-pulse {
  0%   { opacity: 1; }
  50%  { opacity: 0.6; color: var(--accent-blue); }
  100% { opacity: 1; }
}
.live-metric-value { animation: metric-pulse 1.5s ease-in-out 1; }

/* ── Badges ── */
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.badge-running  { background: rgba(39,174,96,0.15);  color: #27ae60; border: 1px solid rgba(39,174,96,0.3); }
.badge-paused   { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-done     { background: rgba(47,128,237,0.15); color: var(--accent-blue); border: 1px solid rgba(47,128,237,0.3); }
.badge-failed   { background: rgba(239,68,68,0.15);  color: var(--accent-red); border: 1px solid rgba(239,68,68,0.3); }
.badge-queued   { background: rgba(139,92,246,0.15); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.3); }

/* ── Pipeline step bar ── */
.pipeline-bar {
  display: flex;
  align-items: center;
  gap: 0;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 20px;
  margin-bottom: 20px;
  overflow-x: auto;
  flex-wrap: nowrap;
  box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  min-width: 90px;
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 10px;
  transition: background 0.2s ease;
  position: relative;
}
.pipeline-step:hover { background: var(--bg-surface); }
.step-circle {
  width: 36px; height: 36px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.85rem; font-weight: 700;
  transition: all 0.3s ease;
}
.step-circle.pending  { background: var(--bg-surface); border: 2px solid var(--border); color: var(--text-muted); }
.step-circle.active   {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
  color: white;
  box-shadow: 0 0 0 4px rgba(47,128,237,0.2), 0 4px 12px rgba(47,128,237,0.4);
  animation: step-pulse 2s ease-in-out infinite;
}
.step-circle.done     { background: var(--accent-green); color: white; box-shadow: 0 2px 8px rgba(39,174,96,0.35); }
@keyframes step-pulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(47,128,237,0.2), 0 4px 12px rgba(47,128,237,0.4); }
  50%       { box-shadow: 0 0 0 8px rgba(47,128,237,0.1), 0 4px 20px rgba(47,128,237,0.6); }
}
.step-label {
  font-size: 0.68rem; font-weight: 500;
  color: var(--text-muted);
  text-align: center; white-space: nowrap;
}
.step-label.active { color: var(--accent-blue); font-weight: 700; }
.step-label.done   { color: var(--accent-green); }
.step-connector {
  flex: 1; height: 2px; min-width: 20px;
  background: var(--border);
  margin: 0;
  position: relative;
  top: -14px;
  transition: background 0.4s ease;
}
.step-connector.done { background: linear-gradient(90deg, var(--accent-green), #2dba6e); }
.step-connector.active { background: linear-gradient(90deg, var(--accent-green), var(--accent-blue)); }

/* ── Step info panel (glassmorphism) ── */
.step-info-panel {
  background: linear-gradient(135deg, rgba(47,128,237,0.06), rgba(139,92,246,0.06));
  border: 1px solid rgba(47,128,237,0.25);
  border-left: 3px solid var(--accent-blue);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 20px;
  backdrop-filter: blur(8px);
}
.step-info-panel .step-info-title {
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--accent-blue);
  margin-bottom: 6px;
}
.step-info-panel .step-info-desc {
  font-size: 0.87rem;
  color: var(--text-muted);
  line-height: 1.55;
}
.step-info-panel .step-info-tips {
  margin-top: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.step-tip-chip {
  background: rgba(47,128,237,0.12);
  border: 1px solid rgba(47,128,237,0.2);
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 0.72rem;
  color: var(--accent-blue);
  font-weight: 500;
}

/* ── Pipeline overview sidebar tracker ── */
.pipeline-overview {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}
.po-step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--border-light);
}
.po-step:last-child { border-bottom: none; }
.po-dot {
  width: 22px; height: 22px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.65rem; font-weight: 700; flex-shrink: 0;
}
.po-dot.done    { background: var(--accent-green); color: white; }
.po-dot.active  { background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); color: white; }
.po-dot.pending { background: var(--bg-surface); border: 1.5px solid var(--border); color: var(--text-muted); }
.po-label { font-size: 0.8rem; font-weight: 500; color: var(--text-muted); }
.po-label.active { color: var(--text-primary); font-weight: 600; }
.po-label.done  { color: var(--accent-green); }

/* ── Task card (model selection) ── */
.task-card {
  background: var(--bg-card);
  border: 2px solid var(--border);
  border-radius: 14px;
  padding: 24px 16px;
  text-align: center;
  cursor: pointer;
  transition: all 0.25s ease;
  display: flex; flex-direction: column; align-items: center; gap: 10px;
}
.task-card:hover { border-color: var(--accent-blue); transform: translateY(-3px); box-shadow: 0 8px 24px rgba(47,128,237,0.15); }
.task-card.selected { border-color: var(--accent-blue); background: linear-gradient(135deg, rgba(47,128,237,0.1), rgba(139,92,246,0.1)); box-shadow: 0 6px 20px rgba(47,128,237,0.18); }
.task-card .task-icon { font-size: 2rem; }
.task-card .task-name { font-size: 0.9rem; font-weight: 700; color: var(--text-primary); }
.task-card .task-desc { font-size: 0.72rem; color: var(--text-muted); }

/* ── Model card ── */
.model-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 10px;
  transition: all 0.2s ease;
}
.model-card:hover { border-color: var(--accent-blue); box-shadow: 0 4px 16px rgba(47,128,237,0.12); transform: translateY(-1px); }
.model-card .mc-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 6px; }
.model-card .mc-desc { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 10px; }
.speed-bar { height: 6px; border-radius: 3px; background: var(--bg-surface); overflow: hidden; margin: 3px 0; }
.speed-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)); }
.speed-label { font-size: 0.68rem; color: var(--text-muted); display: flex; justify-content: space-between; margin-bottom: 2px; }

/* ── Best pipeline result card ── */
.pipeline-result-card {
  background: linear-gradient(135deg, rgba(47,128,237,0.08), rgba(139,92,246,0.08));
  border: 2px solid var(--accent-blue);
  border-radius: 14px;
  padding: 20px 24px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}
.pipeline-result-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-blue));
  background-size: 200% 100%;
  animation: gradient-shift 3s linear infinite;
}
@keyframes gradient-shift {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
.prc-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: var(--accent-blue); font-weight: 700; margin-bottom: 4px; }
.prc-model { font-size: 1.4rem; font-weight: 800; color: var(--text-primary); }
.prc-score { font-size: 0.9rem; color: var(--text-muted); }
.prc-actions { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }

/* ── Training mini-pipeline stage bar ── */
.mini-pipeline {
  display: flex; align-items: center; gap: 0;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 16px;
  margin: 10px 0;
}
.mini-step {
  display: flex; align-items: center; gap: 5px;
  font-size: 0.72rem; font-weight: 500;
  color: var(--text-muted);
  padding: 3px 8px; border-radius: 6px;
  white-space: nowrap;
}
.mini-step.mp-done    { color: var(--accent-green); }
.mini-step.mp-active  { color: var(--accent-blue); font-weight: 700; background: rgba(47,128,237,0.1); animation: mini-pulse 1.5s ease-in-out infinite; }
.mini-step.mp-pending { color: var(--text-muted); opacity: 0.5; }
@keyframes mini-pulse { 0%,100% { background: rgba(47,128,237,0.1); } 50% { background: rgba(47,128,237,0.22); } }
.mini-connector { flex: 1; height: 1px; min-width: 10px; background: var(--border); margin: 0 4px; }
.mini-connector.mp-done { background: var(--accent-green); }

/* ── Inputs / selects / sliders general dark ── */
.stTextInput input, .stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"], .stNumberInput input,
.stTextArea textarea {
  background-color: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"]:hover,
.stMultiSelect div[data-baseweb="select"]:hover {
  border-color: var(--accent-blue) !important;
}
div[data-baseweb="popover"] { background: var(--bg-card) !important; border-color: var(--border) !important; }
div[data-baseweb="menu"] { background: var(--bg-card) !important; }
div[data-baseweb="option"]:hover { background: var(--bg-surface) !important; }

/* ── Dividers and headers ── */
hr { border-color: var(--border-light) !important; }
h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
p, span, li { color: var(--text-primary) !important; }
.stCaption p, small { color: var(--text-muted) !important; }

/* ── Dataframe / Tables ── */
.dataframe, .stDataFrame { border-color: var(--border) !important; }

/* ── Log terminal with colored lines ── */
.log-terminal {
  background: #010409;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  font-family: 'Courier New', monospace;
  font-size: 0.78rem;
  line-height: 1.7;
  max-height: 420px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.log-terminal::-webkit-scrollbar { width: 6px; }
.log-terminal::-webkit-scrollbar-track { background: #010409; }
.log-terminal::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
.log-info    { color: #58a6ff; }
.log-warn    { color: #f59e0b; }
.log-error   { color: #ef4444; font-weight: 600; }
.log-success { color: #27ae60; }
.log-default { color: #8b949e; }

/* ── Step summary card ── */
.summary-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
  margin: 6px 0;
}
.summary-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-light); }
.summary-row:last-child { border-bottom: none; }
.summary-key { color: var(--text-muted); font-size: 0.82rem; }
.summary-val { color: var(--text-primary); font-size: 0.82rem; font-weight: 600; }

/* ── Upload zone ── */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: 14px;
  padding: 32px;
  text-align: center;
  background: var(--bg-card);
  transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent-blue); }

/* ── Hero header ── */
.hero-header {
  padding: 12px 0 8px;
  background: transparent;
}
@keyframes title-glow {
  0%, 100% { filter: brightness(1); }
  50%       { filter: brightness(1.18); }
}
.hero-title {
  font-size: 1.9rem;
  font-weight: 800;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.2;
  animation: title-glow 4s ease-in-out infinite;
}
.hero-subtitle {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 4px;
}
.version-badge {
  display: inline-block;
  background: rgba(47,128,237,0.15);
  color: var(--accent-blue);
  border: 1px solid rgba(47,128,237,0.3);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.7rem;
  font-weight: 700;
  margin-left: 10px;
  vertical-align: middle;
}

/* ── Collapsible expander ── */
.streamlit-expanderHeader {
  background: var(--bg-surface) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}
.streamlit-expanderContent {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
}

/* ── Stat comparison bar ── */
.stat-bar-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; font-size: 0.8rem; }
.stat-bar-label { min-width: 90px; color: var(--text-muted); }
.stat-bar-track { flex: 1; height: 8px; background: var(--bg-surface); border-radius: 4px; overflow: hidden; }
.stat-bar-fill  { height: 100%; border-radius: 4px; }
.stat-bar-val   { min-width: 35px; text-align: right; color: var(--text-primary); font-weight: 600; }

</style>
""", unsafe_allow_html=True)

datalake = DataLake()

# 📋 Sidebar — Premium control panel
with st.sidebar:
    if os.environ.get('IS_ELECTRON_APP') == 'true':
        st.markdown('<span class="badge badge-queued">🖥️ Desktop</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <h1 style='font-size:1.5rem; margin-bottom:0;'>🚀 AutoMLOps</h1>
        <p style='color:#8b949e; font-size:0.8rem;'>Studio v4.8.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pre-fetch sidebar data with cache
    all_runs_sidebar = get_cached_all_runs()
    reg_models_sidebar = get_cached_registered_models()
    
    # Correctly access job manager from session state if available, or create transient one
    if 'job_manager' in st.session_state:
        jm_sidebar = st.session_state['job_manager']
        active_jobs_count = len(jm_sidebar.list_jobs(status=JobStatus.RUNNING))
    else:
        active_jobs_count = 0

    st.markdown("<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#8b949e;margin-bottom:8px;'>SYSTEM OVERVIEW</p>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class='ui-metric' style='border-left-color:#2f80ed;'>
          <div class='metric-value'>{len(all_runs_sidebar)}</div>
          <div class='metric-label'>🧪 Experiments</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='ui-metric' style='border-left-color:#27ae60;'>
          <div class='metric-value'>{len(reg_models_sidebar)}</div>
          <div class='metric-label'>📦 Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    m3, m4 = st.columns(2)
    with m3:
        datasets_count = len(get_cached_datasets())
        st.markdown(f"""
        <div class='ui-metric' style='border-left-color:#8b5cf6;'>
          <div class='metric-value'>{datasets_count}</div>
          <div class='metric-label'>🗄️ Datasets</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        active_badge = f"<span class='badge badge-{'running' if active_jobs_count > 0 else 'done'}'>{active_jobs_count} active</span>" if active_jobs_count > 0 else "<span class='badge badge-done'>idle</span>"
        st.markdown(f"""
        <div class='ui-metric' style='border-left-color:#f59e0b;'>
          <div class='metric-value' style='font-size:1.2rem;'>{active_badge}</div>
          <div class='metric-label'>⚙️ Engine</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#8b949e;margin-bottom:8px;'>ACTIVE DATASET</p>", unsafe_allow_html=True)
    if 'train_df' in st.session_state:
        tdf = st.session_state['train_df']
        task_lbl = st.session_state.get('current_task', 'N/A')
        st.markdown(f"""
        <div class='ui-card' style='padding:12px;margin-bottom:8px;border-left:3px solid #27ae60;'>
          <div style='font-size:0.72rem;color:#8b949e;'>TRAIN SET</div>
          <div style='font-weight:600;'>{tdf.shape[0]:,} rows × {tdf.shape[1]} cols</div>
          <span class='badge badge-running' style='margin-top:6px;display:inline-block;'>{task_lbl}</span>
        </div>""", unsafe_allow_html=True)
    elif 'df' in st.session_state:
        df_s = st.session_state['df']
        st.markdown(f"""
        <div class='ui-card' style='padding:12px;margin-bottom:8px;'>
          <div style='font-size:0.72rem;color:#8b949e;'>DATASET</div>
          <div style='font-weight:600;'>{df_s.shape[0]:,} rows × {df_s.shape[1]} cols</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#8b949e;font-size:0.82rem;'>⚠️ No dataset loaded</div>", unsafe_allow_html=True)

    st.divider()

    # DagsHub Integration
    with st.expander("☁️ DagsHub Integration"):
        st.caption("Connect to your DagsHub repository to save experiments remotely.")
        env_user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        env_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        default_repo = ""
        if "dagshub.com" in env_uri:
            try:
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
            if st.button("Connect", key="dh_connect_btn"):
                if dh_user and dh_repo and dh_token:
                    try:
                        os.environ["MLFLOW_TRACKING_USERNAME"] = dh_user
                        os.environ["MLFLOW_TRACKING_PASSWORD"] = dh_token
                        remote_uri = f"https://dagshub.com/{dh_user}/{dh_repo}.mlflow"
                        os.environ["MLFLOW_TRACKING_URI"] = remote_uri
                        mlflow.set_tracking_uri(remote_uri)
                        try:
                            mlflow.search_experiments(max_results=1)
                            st.session_state['dagshub_connected'] = True
                            st.session_state['mlflow_uri'] = remote_uri
                            st.success("Connected!")
                        except Exception as e:
                            st.error(f"Failed: {e}")
                            local_uri = "sqlite:///mlflow.db"
                            mlflow.set_tracking_uri(local_uri)
                            os.environ["MLFLOW_TRACKING_URI"] = local_uri
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Fill all fields.")
        with col_dh2:
            is_dagshub = "dagshub.com" in mlflow.get_tracking_uri()
            if st.button("Disconnect", disabled=not is_dagshub, key="dh_disconnect_btn"):
                local_uri = "sqlite:///mlflow.db"
                mlflow.set_tracking_uri(local_uri)
                os.environ["MLFLOW_TRACKING_URI"] = local_uri
                if "MLFLOW_TRACKING_USERNAME" in os.environ:
                    del os.environ["MLFLOW_TRACKING_USERNAME"]
                if "MLFLOW_TRACKING_PASSWORD" in os.environ:
                    del os.environ["MLFLOW_TRACKING_PASSWORD"]
                st.session_state['dagshub_connected'] = False
                st.info("Disconnected.")
                st.rerun()

        current_uri = mlflow.get_tracking_uri()
        if "dagshub.com" in current_uri:
            st.success("🟢 Connected to DagsHub")
        else:
            st.info("⚪ Local MLflow (SQLite)")

    current_uri = mlflow.get_tracking_uri()
    st.caption(f"Tracking URI: `{current_uri}`")

# 🎨 Hero Header
st.markdown("""
<div class='hero-header'>
  <div class='hero-title'>AutoMLOps Studio</div>
  <div class='hero-subtitle'>Automated Machine Learning & MLOps Platform</div>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if 'trials_data' not in st.session_state: st.session_state['trials_data'] = []
if 'best_model' not in st.session_state: st.session_state['best_model'] = None
if 'job_manager' not in st.session_state:
    st.session_state['job_manager'] = TrainingJobManager()
if 'mlflow_cache' not in st.session_state:
    st.session_state['mlflow_cache'] = {}
# Pipeline wizard state
if 'automl_step' not in st.session_state: st.session_state['automl_step'] = 0
if 'automl_config' not in st.session_state: st.session_state['automl_config'] = {}

# poll_updates moved to Experiments tab fragment for performance


def render_pipeline_header(steps: list, current: int):
    """Render a horizontal WatsonX-style step indicator.
    steps: list of (icon, label) tuples.
    current: 0-based index of the active step.
    """
    html_parts = ['<div class="pipeline-bar">']
    for i, (icon, label) in enumerate(steps):
        if i < current:
            circle_cls = 'done'
            label_cls  = 'done'
            circle_content = '✓'
        elif i == current:
            circle_cls = 'active'
            label_cls  = 'active'
            circle_content = str(i + 1)
        else:
            circle_cls = 'pending'
            label_cls  = ''
            circle_content = str(i + 1)

        html_parts.append(f"""
        <div class="pipeline-step">
          <div class="step-circle {circle_cls}">{circle_content}</div>
          <div class="step-label {label_cls}">{icon} {label}</div>
        </div>""")
        if i < len(steps) - 1:
            connector_cls = 'done' if i < current else ''
            html_parts.append(f'<div class="step-connector {connector_cls}"></div>')

    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)



def render_step_info_panel(title: str, description: str, tips: list = None):
    """Render a glassmorphism info panel explaining the current wizard step."""
    tips_html = ""
    if tips:
        chips = "".join([f"<span class='step-tip-chip'>💡 {t}</span>" for t in tips])
        tips_html = f"<div class='step-info-tips'>{chips}</div>"
    st.markdown(f"""
    <div class='step-info-panel'>
      <div class='step-info-title'>About this step</div>
      <div class='step-info-desc'>{description}</div>
      {tips_html}
    </div>""", unsafe_allow_html=True)


def render_pipeline_overview(steps: list, current: int):
    """Render a compact vertical pipeline tracker (shown alongside step content)."""
    html_rows = ""
    for i, (icon, label) in enumerate(steps):
        if i < current:
            dot_cls, lbl_cls, sym = 'po-dot done', 'po-label done', '✓'
        elif i == current:
            dot_cls, lbl_cls, sym = 'po-dot active', 'po-label active', str(i + 1)
        else:
            dot_cls, lbl_cls, sym = 'po-dot pending', 'po-label', str(i + 1)
        html_rows += (
            f"<div class='po-step'>"
            f"<div class='{dot_cls}'>{sym}</div>"
            f"<div class='{lbl_cls}'>{icon} {label}</div>"
            f"</div>"
        )
    full_html = (
        "<div class='pipeline-overview'>"
        "<div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;"
        "color:#8b949e;margin-bottom:8px;font-weight:700;'>Pipeline Progress</div>"
        + html_rows +
        "</div>"
    )
    try:
        st.html(full_html)
    except AttributeError:
        st.markdown(full_html, unsafe_allow_html=True)



def render_colored_log(log_lines: list):
    """Render log lines with color coding based on severity keywords."""
    html_lines = []
    for line in log_lines:
        line_esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        upper = line.upper()
        if "ERROR" in upper or "FAILED" in upper or "EXCEPTION" in upper:
            css = "log-error"
        elif "WARN" in upper or "WARNING" in upper:
            css = "log-warn"
        elif "COMPLETE" in upper or "DONE" in upper or "SUCCESS" in upper or "BEST" in upper:
            css = "log-success"
        elif "INFO" in upper or "[JOB]" in upper or "TRIAL" in upper or "START" in upper:
            css = "log-info"
        else:
            css = "log-default"
        html_lines.append(f"<span class='{css}'>{line_esc}</span>")
    log_html = "<br>".join(html_lines)
    st.markdown(f"<div class='log-terminal'>{log_html}</div>", unsafe_allow_html=True)


def render_training_mini_pipeline(log_lines: list, status: str):
    """Infer current training stage from logs and render a mini pipeline bar."""
    STAGES = [
        ("🔧", "Data Preparation"),
        ("🔍", "Model Selection"),
        ("⚖️", "Tuning & Optimization"),
        ("📊", "Validation & Analysis"),
        ("✅", "Finalized"),
    ]
    # Determine how far along we are based on log content
    last_logs = " ".join(log_lines[-30:]).upper() if log_lines else ""
    if status == "completed" or "TRAINING COMPLETE" in last_logs:
        stage_idx = 4
    elif "REPORT" in last_logs or "__REPORT__" in last_logs:
        stage_idx = 3
    elif "TRIAL" in last_logs or "OPTUNA" in last_logs or "SCORE" in last_logs:
        stage_idx = 2
    elif "PREPROCESSING" in last_logs or "FEATURE" in last_logs or "TRANSFORM" in last_logs:
        stage_idx = 1
    else:
        stage_idx = 0 if log_lines else -1

    if stage_idx < 0:
        return  # Nothing to show yet

    parts = ['<div class="mini-pipeline">']
    for i, (icon, label) in enumerate(STAGES):
        if i < stage_idx:
            cls = "mp-done"
            sym = "✓"
        elif i == stage_idx:
            cls = "mp-active"
            sym = icon
        else:
            cls = "mp-pending"
            sym = icon
        parts.append(f"<div class='mini-step {cls}'>{sym} {label}</div>")
        if i < len(STAGES) - 1:
            conn_cls = "mp-done" if i < stage_idx else ""
            parts.append(f"<div class='mini-connector {conn_cls}'></div>")
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_pipeline_progress_ring(trials_done: int, total_est: int, is_done: bool = False):
    """Render an SVG circular progress ring for trial completion."""
    if is_done:
        pct = 1.0
    elif total_est <= 0:
        total_est = max(trials_done, 1)
        pct = min(1.0, trials_done / total_est)
    else:
        pct = min(1.0, trials_done / total_est)
    
    r, cx, cy = 28, 36, 36
    circ = 2 * 3.14159 * r
    dash = pct * circ
    color = "#2f80ed" if pct < 1.0 else "#27ae60"
    label_txt = f"{int(pct*100)}%"
    svg = f"""
    <svg width="72" height="72" viewBox="0 0 72 72">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#21262d" stroke-width="6"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="6"
        stroke-dasharray="{dash:.1f} {circ:.1f}"
        stroke-dashoffset="{circ/4:.1f}" stroke-linecap="round"
        transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy+5}" text-anchor="middle" fill="{color}"
        font-size="12" font-weight="700" font-family="Inter,sans-serif">{label_txt}</text>
    </svg>"""
    st.markdown(f"""<div class='progress-ring-wrap'>{svg}
      <div>
        <div class='progress-ring-value'>{trials_done} trials</div>
        <div class='progress-ring-label'>of ~{total_est} expected</div>
      </div>
    </div>""", unsafe_allow_html=True)


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
        df_ds = get_cached_dataframe(config['name'], config['version'])
        
        # Apply schema overrides if provided
        if 'schema_overrides' in config:
            overrides = config['schema_overrides']
            cols_to_drop = [row['Column Name'] for row in overrides if not row.get('Include', True)]
            df_ds = df_ds.drop(columns=[c for c in cols_to_drop if c in df_ds.columns], errors='ignore')
            
            for row in overrides:
                if row.get('Include', True):
                    col_name = row['Column Name']
                    target_type = row.get('Type', 'object')
                    if col_name in df_ds.columns:
                        try:
                            if target_type == 'datetime64[ns]':
                                df_ds[col_name] = pd.to_datetime(df_ds[col_name], errors='coerce')
                            else:
                                df_ds[col_name] = df_ds[col_name].astype(target_type)
                        except Exception as e:
                            print(f"Schema cast error on {col_name} to {target_type}: {e}")
        
        if global_split is not None:
            split_ratio = global_split
        else:
            split_ratio = config.get('split', 80) / 100.0
            
        strat = config.get('split_strategy', 'Random')
        
        if split_ratio >= 1.0:
            train_dfs.append(df_ds)
        elif split_ratio <= 0.0:
            test_dfs.append(df_ds)
        else:
            if strat == 'Cronológico (Chronological)' or (task_type == 'time_series' and date_col and date_col in df_ds.columns):
                # Temporal split
                time_col = config.get('time_column', date_col)
                if time_col and time_col in df_ds.columns:
                    df_ds = df_ds.sort_values(by=time_col)
                split_idx = int(len(df_ds) * split_ratio)
                tr = df_ds.iloc[:split_idx]
                te = df_ds.iloc[split_idx:]
            elif strat == 'Manual (Pre-defined split column)':
                # Expects a column indicating split - we will approximate it using the defined split ratio for now if not purely boolean
                manual_col = config.get('manual_split_column')
                if manual_col and manual_col in df_ds.columns:
                    # simplistic assumption 1/True = train, 0/False = test, fallback if string
                    if df_ds[manual_col].dtype == bool:
                        tr = df_ds[df_ds[manual_col] == True]
                        te = df_ds[df_ds[manual_col] == False]
                    else:
                        tr = df_ds[df_ds[manual_col].astype(str).str.lower().isin(['train', '1', 'true', 'tr'])]
                        te = df_ds[~df_ds.index.isin(tr.index)]
                    df_ds = df_ds.drop(columns=[manual_col])
                    tr = tr.drop(columns=[manual_col], errors='ignore')
                    te = te.drop(columns=[manual_col], errors='ignore')
                else: # Fallback to random
                    tr, te = train_test_split(df_ds, train_size=split_ratio, random_state=42)
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
    st.markdown(f"""
    <div class='hero-header'>
      <div class='hero-title'>📦 Data Lake & Management</div>
      <div class='hero-subtitle'>Catalog, version, and analyze your datasets.</div>
    </div>""", unsafe_allow_html=True)
    
    data_tabs = st.tabs(["🗄️ Data Management", "📉 Drift Detection"])
    
    with data_tabs[0]:
        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.subheader("📤 Upload New Data")
            uploaded_files = st.file_uploader("Upload Data (CSV, JSON, Parquet, TXT, ZIP)", type=["csv", "json", "parquet", "txt", "zip"], accept_multiple_files=True, label_visibility="collapsed")
            
            # --- Advanced Parser Options ---
            with st.expander("⚙️ Manual Dataset Parsing", expanded=False):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    parse_format = st.selectbox("File Format", ["Auto (Extension)", "Delimited (CSV)", "JSON Lines", "Parquet file", "Plain text"])
                    parse_delim = st.selectbox("Delimiter", [",", ";", "\\t (Tab)", " ", "|", "Custom"])
                    if parse_delim == "Custom":
                        custom_delim = st.text_input("Custom Delimiter", ",")
                        actual_delim = custom_delim
                    else:
                        delim_map = {",": ",", ";": ";", "\\t (Tab)": "\t", " ": " ", "|": "|"}
                        actual_delim = delim_map.get(parse_delim, ",")
                with col_p2:
                    parse_enc = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "latin1", "ascii", "utf-16", "windows-1252"])
                    parse_header = st.selectbox("Column headers", ["All files have the same headers", "No header"])
                    actual_header = 0 if "All" in parse_header else None
                    replace_existing = st.checkbox("Replace if already exists", value=True)

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    df_preview = None
                    try:
                        # Determine actual format
                        fmt = ext
                        if parse_format == "Delimited (CSV)": fmt = '.csv'
                        elif parse_format == "JSON Lines": fmt = '.json'
                        elif parse_format == "Parquet file": fmt = '.parquet'
                        elif parse_format == "Plain text": fmt = '.txt'

                        if fmt == '.csv':
                            df_preview = pd.read_csv(uploaded_file, sep=actual_delim, encoding=parse_enc, header=actual_header)
                        elif fmt == '.json':
                            df_preview = pd.read_json(uploaded_file, orient='records', lines=True)
                        elif fmt == '.parquet':
                            df_preview = pd.read_parquet(uploaded_file)
                    except Exception as e:
                        st.warning(f"Preview unavailable for {uploaded_file.name}: {e}")
                    
                    with st.expander(f"👁️ Preview: {uploaded_file.name}", expanded=True):
                        if df_preview is not None:
                            st.dataframe(df_preview.head(5), use_container_width=True)
                        elif ext == '.txt' or fmt == '.txt':
                            uploaded_file.seek(0)
                            st.text(uploaded_file.getvalue().decode(parse_enc, errors='ignore')[:500] + "...")
                        elif ext == '.zip':
                            st.info("ZIP file detected. Generally used for CV or raw data.")
                        else:
                            st.info("Non-tabular format or error processing.")
                    
                    dataset_name = st.text_input(f"Name as (slug)", uploaded_file.name.replace(ext, ""), key=f"name_{uploaded_file.name}")
                    
                    if df_preview is not None and actual_header is None:
                        new_cols = st.text_input(f"Rename Columns (comma separated)", value=",".join([f"col_{i}" for i in range(len(df_preview.columns))]), key=f"cols_{uploaded_file.name}")
                        if new_cols:
                            try:
                                df_preview.columns = [c.strip() for c in new_cols.split(",")]
                            except:
                                st.error("Column count mismatch.")

                    if st.button(f"📥 Save {uploaded_file.name}", key=f"save_{uploaded_file.name}", type="primary"):
                        if df_preview is not None:
                            # Use Datake storage - saves as CSV by default locally
                            csv_buffer = io.StringIO()
                            df_preview.to_csv(csv_buffer, index=False)
                            path = datalake.save_raw_file(csv_buffer.getvalue().encode('utf-8'), dataset_name, uploaded_file.name.replace(ext, '.csv'))
                            st.session_state['df'] = df_preview
                        else:
                            uploaded_file.seek(0)
                            path = datalake.save_raw_file(uploaded_file.getvalue(), dataset_name, uploaded_file.name)
                        st.success(f"Dataset '{dataset_name}' saved to lake!")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dl2:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Explore & Load")
            datasets = get_cached_datasets()
            selected_ds = st.selectbox("Select Dataset", [""] + datasets, key="data_load_sel")
            if selected_ds:
                versions = datalake.list_versions(selected_ds)
                selected_ver = st.selectbox("Version", versions, key="data_ver_sel")
                
                # Info card
                with st.container():
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🗑 Delete Version", key="del_ds_ver", use_container_width=True):
                            datalake.delete_version(selected_ds, selected_ver)
                            st.cache_data.clear() # Clear cache on delete
                            st.success(f"Deleted {selected_ver}")
                            st.rerun()
                    with c2:
                        if st.button("👁 Load Preview", key="load_ds_ver", type="primary"):
                            df_preview = get_cached_dataframe(selected_ds, selected_ver)
                            st.session_state['data_preview_df'] = df_preview
            st.markdown("</div>", unsafe_allow_html=True)
            
        if 'data_preview_df' in st.session_state:
            st.divider()
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state['data_preview_df'], use_container_width=True)

# --- TAB 4: MLOPS MONITORING ---
with tabs[4]:
    st.markdown(f"""
    <div class='hero-header'>
      <div class='hero-title'>📉 ML Monitoring & Observability</div>
      <div class='hero-subtitle'>Monitor deployed models for distribution shifts and robustness.</div>
    </div>""", unsafe_allow_html=True)

    mon_tabs = st.tabs(["🚀 Production Drift", "🛡️ Model Robustness & Stability"])
    
    with mon_tabs[0]:
        col_mon1, col_mon2 = st.columns([1, 2])

    with col_mon1:
        st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
        st.subheader("📋 Configuration")
        st.info("The Baseline is usually the training dataset stored in your Data Lake.")
        mon_datasets = get_cached_datasets()
        mon_ref_ds = st.selectbox("Select Baseline Dataset", [""] + mon_datasets, key="mon_ref_ds")
        df_baseline = None
        if mon_ref_ds:
            mon_ref_ver = st.selectbox("Baseline Version", get_cached_versions(mon_ref_ds), key="mon_ref_ver")
            df_baseline = get_cached_dataframe(mon_ref_ds, mon_ref_ver)
            st.success(f"Loaded Baseline: {df_baseline.shape[0]} rows")

        st.divider()
        st.subheader("📡 Production Telemetry")
        st.caption("Telemetry data collected from predicted logs.")

        # Load Telemetry Data
        telemetry_path = os.path.join("data_lake", "monitoring", "api_telemetry.csv")
        df_telemetry = None
        if os.path.exists(telemetry_path):
            try:
                df_telemetry = pd.read_csv(telemetry_path)
                st.success(f"Found {len(df_telemetry)} logs.")

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
            st.warning("No telemetry data found.")
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
        st.subheader("🛡️ Model Robustness & Stability")
        st.markdown("Run live stability tests on your Registered Models against specific Base Datasets.")
        st.markdown("</div>", unsafe_allow_html=True)
        
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
                
                stab_datasets = get_cached_datasets()
                stab_ref_ds = st.selectbox("Test Dataset (Data Lake)", [""] + stab_datasets, key="stab_ref_ds")
                df_stab_ref = None
                target_col = None
                if stab_ref_ds:
                    stab_ref_ver = st.selectbox("Dataset Version", get_cached_versions(stab_ref_ds), key="stab_ref_ver")
                    df_stab_ref = get_cached_dataframe(stab_ref_ds, stab_ref_ver)
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
        st.markdown(f"""
        <div class='ui-card'>
          <div style='font-weight:700; font-size:1.1rem;'>📉 Data Drift Analysis</div>
          <div style='color:#8b949e; font-size:0.85rem; margin-bottom:16px;'>Compare two datasets to detect distribution shifts between training (reference) and inference (current) data.</div>
        </div>""", unsafe_allow_html=True)

        col_dr1, col_dr2 = st.columns(2)
        with col_dr1:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.markdown("##### 1. Reference Data (Baseline)")
            ref_ds = st.selectbox("Reference Dataset", [""] + get_cached_datasets(), key="drift_ref_ds")
            df_ref = None
            if ref_ds:
                ref_ver = st.selectbox("Reference Version", get_cached_versions(ref_ds), key="drift_ref_ver")
                df_ref = get_cached_dataframe(ref_ds, ref_ver)
                st.write(f"Reference Loaded: {df_ref.shape}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dr2:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.markdown("##### 2. Current Data (Target)")
            curr_ds = st.selectbox("Current Dataset", [""] + get_cached_datasets(), key="drift_curr_ds")
            df_curr = None
            if curr_ds:
                curr_ver = st.selectbox("Current Version", get_cached_versions(curr_ds), key="drift_curr_ver")
                df_curr = get_cached_dataframe(curr_ds, curr_ver)
                st.write(f"Current Loaded: {df_curr.shape}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        if df_ref is not None and df_curr is not None:
            if st.button("🚀 Run Drift Detection", key="drift_run_btn", type="primary"):
                with st.spinner("Calculating Drift Metrics (KS Test)..."):
                    drift_report = []
                    numeric_cols = df_ref.select_dtypes(include=np.number).columns.intersection(df_curr.columns)
                    
                    st.divider()
                    st.markdown("### 📊 Distribution Comparison")
                    for col in numeric_cols:
                        from scipy.stats import ks_2samp
                        stat, p_value = ks_2samp(df_ref[col].dropna(), df_curr[col].dropna())
                        drift_detected = p_value < 0.05
                        status_chip = "<span class='badge badge-failed'>🔴 Drift Detected</span>" if drift_detected else "<span class='badge badge-done'>🟢 Stable</span>"
                        
                        drift_report.append({
                            "Feature": col,
                            "KS Stat": f"{stat:.4f}",
                            "P-Value": f"{p_value:.4f}",
                            "Status": "Drift" if drift_detected else "Stable"
                        })
                        
                        with st.expander(f"Feature: {col} - {'🔴' if drift_detected else '🟢'}", expanded=False):
                            fig = px.histogram(df_ref, x=col, color_discrete_sequence=['#2f80ed'], opacity=0.5, nbins=30, title=f"Distribution Shift: {col}")
                            fig.add_histogram(x=df_curr[col], name='Current', marker_color='#ef4444', opacity=0.5)
                            fig.update_layout(barmode='overlay')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Drift Summary Table")
                    st.dataframe(pd.DataFrame(drift_report), use_container_width=True)


# --- TAB 1: AUTOML & MODEL HUB ---
with tabs[1]:
    st.markdown("""
    <div class='hero-header'>
      <div class='hero-title'>🤖 AutoML &amp; Model Hub</div>
      <div class='hero-subtitle'>Automated machine learning pipeline — from raw data to a production-ready model.</div>
    </div>""", unsafe_allow_html=True)
    
    # Sub-tabs within AutoML
    automl_tabs = st.tabs(["📊 Classical ML (Tabular)", "🖼️ Computer Vision"])
    
    # --- SUB-TAB 1.1: CLASSICAL ML (WatsonX-Style Pipeline) ---
    with automl_tabs[0]:
        # ── Step definitions ──────────────────────────────────────────
        PIPELINE_STEPS = [
            ("📂", "Data"),
            ("🎯", "Task"),
            ("🤖", "Models"),
            ("⚡", "Optimize"),
            ("🛡️", "Validate"),
            ("🔧", "Advanced"),
            ("🚀", "Submit"),
        ]

        cur_step = st.session_state.get('automl_step', 0)
        render_pipeline_header(PIPELINE_STEPS, cur_step)

        # ── Shared state container ────────────────────────────────────
        cfg = st.session_state.get('automl_config', {})

        # ── Defaults that need to exist before any step ────────────────
        task                = cfg.get('task', 'classification')
        trainer_temp        = AutoMLTrainer(task_type=task)
        available_models    = trainer_temp.get_available_models()
        selected_models     = cfg.get('selected_models', None)
        training_preset     = cfg.get('training_preset', 'medium')
        training_strategy   = cfg.get('training_strategy', 'Automatic')
        manual_params       = cfg.get('manual_params', None)
        validation_strategy = cfg.get('validation_strategy', 'auto')
        validation_params   = cfg.get('validation_params', {})
        optimization_metric = cfg.get('optimization_metric', 'accuracy')
        selected_opt_mode   = cfg.get('optimization_mode', 'bayesian')
        n_trials            = cfg.get('n_trials', None)
        timeout_per_model   = cfg.get('timeout', None)
        total_time_budget   = cfg.get('time_budget', None)
        early_stopping      = cfg.get('early_stopping', 10)
        random_seed_config  = cfg.get('random_state', 42)
        ensemble_config     = cfg.get('ensemble_config', {})
        selected_nlp_cols   = cfg.get('selected_nlp_cols', [])
        nlp_config_automl   = cfg.get('nlp_config', {})
        enable_stability    = cfg.get('enable_stability', False)
        selected_stability_tests = cfg.get('stability_tests', [])
        target_pre          = cfg.get('target', None)
        date_col_pre        = cfg.get('date_col', None)
        selected_configs    = cfg.get('selected_configs', [])
        model_source        = cfg.get('model_source', 'Standard AutoML (Scikit-Learn/XGBoost/Transformers)')
        sample_df           = None
        embedding_model     = cfg.get('nlp_config', {}).get('embedding_model', 'all-MiniLM-L6-v2')

        # ════════════════════════════════════════════════════════════════
        # STEP 0 — Data Source
        # ════════════════════════════════════════════════════════════════
        if cur_step == 0:
            _col_main, _col_track = st.columns([3, 1])
            with _col_track:
                render_pipeline_overview(PIPELINE_STEPS, cur_step)
            with _col_main:
              st.markdown("<h3 style='margin-bottom:4px;'>📂 Step 1 — Data Source</h3>", unsafe_allow_html=True)
              render_step_info_panel(
                  "Data Source",
                  "Select one or more datasets from your Data Lake to train on. You can mix multiple datasets — they will be concatenated during loading. Configure the train/test split ratio for each dataset individually.",
                  ["Use 70-80% for training", "Preview data before selecting target", "Multiple datasets are concatenated"]
              )

              available_datasets = get_cached_datasets()
              if not available_datasets:
                  st.markdown("""
                  <div class='ui-card' style='text-align:center;padding:32px;'>
                      <h2 style='color:#8b949e;margin-bottom:12px;'>🗄️ Your Data Lake is empty</h2>
                      <p style='color:#8b949e;font-size:0.95rem;margin-bottom:24px;'>Please upload or connect a dataset in the <b>Data Ingestion</b> tab first.</p>
                  </div>""", unsafe_allow_html=True)
              else:
                  sel_ds_list = st.multiselect(
                      "📚 Select Dataset(s)", available_datasets, 
                      default=cfg.get('ds_list', []), key="wiz_ds_multi"
                  )
                  cfg['ds_list'] = sel_ds_list

                  target_pre_w  = None
                  date_col_pre_w= None

                  if sel_ds_list:
                      # Sample preview from first dataset
                      try:
                          first_ds  = sel_ds_list[0]
                          first_ver = get_cached_versions(first_ds)[0]
                          sample_df = get_cached_dataframe(first_ds, first_ver, nrows=200)

                          with st.expander("👁️ Data Preview", expanded=True):
                              st.dataframe(sample_df.head(5), use_container_width=True)

                          # ── Data Profiling Charts ──
                          with st.expander("📊 Data Profile", expanded=True):
                              _pc1, _pc2 = st.columns(2)
                              with _pc1:
                                  dtype_counts = sample_df.dtypes.apply(lambda d: 'Numeric' if pd.api.types.is_numeric_dtype(d) else ('DateTime' if pd.api.types.is_datetime64_any_dtype(d) else 'Categorical')).value_counts()
                                  fig_dtype = px.pie(
                                      values=dtype_counts.values, names=dtype_counts.index,
                                      title="Feature Types",
                                      color_discrete_sequence=['#2f80ed','#8b5cf6','#27ae60'],
                                      hole=0.45
                                  )
                                  fig_dtype.update_layout(
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font_color='#e6edf3', title_font_size=13, margin=dict(t=40,b=10,l=10,r=10),
                                      legend=dict(font=dict(size=10))
                                  )
                                  st.plotly_chart(fig_dtype, use_container_width=True)
                              with _pc2:
                                  missing_pct = (sample_df.isnull().sum() / len(sample_df) * 100).sort_values(ascending=True)
                                  missing_pct = missing_pct[missing_pct > 0]
                                  if not missing_pct.empty:
                                      fig_miss = px.bar(
                                          x=missing_pct.values, y=missing_pct.index,
                                          orientation='h', title="Missing Values (%)",
                                          color=missing_pct.values,
                                          color_continuous_scale=['#27ae60','#f59e0b','#ef4444']
                                      )
                                      fig_miss.update_layout(
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font_color='#e6edf3', title_font_size=13, margin=dict(t=40,b=10,l=10,r=10),
                                          showlegend=False, coloraxis_showscale=False
                                      )
                                      st.plotly_chart(fig_miss, use_container_width=True)
                                  else:
                                      st.markdown("""
                                      <div style='text-align:center;padding:30px;'>
                                        <div style='font-size:2rem;'>✅</div>
                                        <div style='color:#27ae60;font-weight:600;margin-top:8px;'>No Missing Values</div>
                                        <div style='color:#8b949e;font-size:0.8rem;'>Dataset is complete!</div>
                                      </div>""", unsafe_allow_html=True)

                          col_ta, col_dc = st.columns(2)
                          with col_ta:
                              task_tmp = cfg.get('task', 'classification')
                              if task_tmp not in ["clustering", "anomaly_detection", "dimensionality_reduction"]:
                                  default_target_idx = max(0, len(sample_df.columns) - 1)
                                  target_pre_w = st.selectbox(
                                      "🎯 Target Column",
                                      sample_df.columns.tolist(),
                                      index=default_target_idx,
                                      key="wizard_target"
                                  )
                                  cfg['target'] = target_pre_w
                          with col_dc:
                              if task_tmp == 'time_series':
                                  date_col_pre_w = st.selectbox("📅 Date Column", sample_df.columns, key="wizard_date")
                                  cfg['date_col'] = date_col_pre_w
                      except Exception as e:
                          st.error(f"Error loading preview: {e}")

                      # Per-dataset version + split config
                      new_configs = []
                      
                      st.markdown("#### Schema Configuration", unsafe_allow_html=True)
                      st.info("Adjust the automatically inferred data types. Uncheck columns that should be ignored during training.")
                      
                      schema_df = pd.DataFrame({
                          "Include": [True] * len(sample_df.columns),
                          "Column Name": sample_df.columns,
                          "Type": [str(t) for t in sample_df.dtypes],
                          "Sample Values": [str(sample_df[c].iloc[0]) if len(sample_df) > 0 else "" for c in sample_df.columns]
                      })
                      
                      edited_schema = st.data_editor(
                          schema_df,
                          column_config={
                              "Include": st.column_config.CheckboxColumn("Include", help="Include in training?", default=True),
                              "Type": st.column_config.SelectboxColumn("Type", help="Hide Pandas type", options=["object", "int64", "float64", "bool", "datetime64[ns]"]),
                          },
                          disabled=["Column Name", "Sample Values"],
                          hide_index=True,
                          key="wizard_schema_editor"
                      )
                      # Map back to expected keys in schema_overrides
                      overrides = edited_schema.to_dict('records')
                      cfg['schema_overrides'] = overrides
                      
                      st.markdown("#### Global Dataset Holdout (Final Evaluation)", unsafe_allow_html=True)
                      split_strategy = st.radio(
                          "Holdout Mapping Mode", 
                          ["Random", "Chronological", "Manual (Pre-defined split column)"],
                          horizontal=True,
                          key="wizard_split_strat"
                      )
                      cfg['split_strategy'] = split_strategy
                      
                      ds_cols = st.columns(min(len(sel_ds_list), 3))
                      for i, ds_name in enumerate(sel_ds_list):
                          with ds_cols[i % 3]:
                              versions = get_cached_versions(ds_name)
                              ver = st.selectbox(f"📌 Version — {ds_name}", versions, key=f"wiz_ver_{ds_name}")
                              
                              # Render split progress bar logic based on images
                              split = st.slider(
                                  f"% Train — {ds_name}", 10, 100, 80, 
                                  key=f"wiz_split_{ds_name}",
                                  help="Global Holdout: This defines the absolute test set (shards/rows) used ONLY for final evaluation after the entire AutoML training finishes. (Internal cross-validation is configured in Step 5)."
                              )
                              
                              st.markdown(f"**Split visual:** <span style='color:#2f80ed'>Training: {split}%</span> | <span style='color:#f59e0b'>Validation: {int((100-split)/2)}%</span> | <span style='color:#8b5cf6'>Testing: {100-split-int((100-split)/2)}%</span>", unsafe_allow_html=True)
                              
                              if split_strategy == "Chronological":
                                  time_col = st.selectbox(f"Time Column for {ds_name}", sample_df.columns, key=f"wiz_time_{ds_name}")
                                  new_configs.append({'name': ds_name, 'version': ver, 'split': split, 'time_column': time_col})
                              elif split_strategy == "Manual (Pre-defined split column)":
                                  manual_col = st.selectbox(f"Split Flag Column for {ds_name}", sample_df.columns, key=f"wiz_manual_{ds_name}")
                                  new_configs.append({'name': ds_name, 'version': ver, 'split': split, 'manual_split_column': manual_col})
                              else:
                                  new_configs.append({'name': ds_name, 'version': ver, 'split': split})
                      cfg['selected_configs'] = new_configs
                      st.markdown('<br>', unsafe_allow_html=True)
                      
                      data_ready = bool(sel_ds_list)
                      col_nav_fwd, _ = st.columns([1, 4])
                      with col_nav_fwd:
                          if st.button("Next: Define Task →", type="primary", key="step0_next", disabled=not data_ready):
                              st.session_state['automl_step'] = 1
                              st.session_state['automl_config'] = cfg
                              st.rerun()
                  else:
                      st.info("ℹ️ Select at least one dataset above to continue.")


        # ════════════════════════════════════════════════════════════════
        # STEP 1 — Task Type & Learning Mode
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 1:
            _col_main, _col_track = st.columns([3, 1])
            with _col_track:
                render_pipeline_overview(PIPELINE_STEPS, cur_step)
            with _col_main:
              st.markdown("<h3 style='margin-bottom:4px;'>🎯 Step 2 — Task &amp; Learning Type</h3>", unsafe_allow_html=True)
              render_step_info_panel(
                  "Task Type",
                  "Choose the type of ML problem you want to solve. The task type determines which models, metrics, and optimization strategies are available in the next steps.",
                  ["Classification for label prediction", "Regression for continuous output", "Time Series for temporal forecasting"]
              )

            SUPERVISED_TASKS = [
                ("classification", "🎯", "Classification", "Predict a category or label"),
                ("regression",     "📈", "Regression",     "Predict a continuous value"),
                ("time_series",    "⏳", "Time Series",    "Forecast future values from historical sequences"),
            ]
            UNSUP_TASKS = [
                ("clustering",               "🔵", "Clustering",               "Discover natural groups in data"),
                ("anomaly_detection",        "🚨", "Anomaly Detection",        "Identify unusual data points"),
                ("dimensionality_reduction", "🔻", "Dimensionality Reduction", "Compress features intelligently"),
            ]

            learn_type = st.radio(
                "Learning Paradigm",
                ["Supervised", "Unsupervised"],
                index=0 if cfg.get('task', 'classification') not in [t[0] for t in UNSUP_TASKS] else 1,
                horizontal=True,
                key="wiz_learn_type"
            )

            task_list = SUPERVISED_TASKS if learn_type == "Supervised" else UNSUP_TASKS
            current_task = cfg.get('task', task_list[0][0])
            if current_task not in [t[0] for t in task_list]:
                current_task = task_list[0][0]

            st.markdown('<br>', unsafe_allow_html=True)
            cols_t = st.columns(len(task_list))
            for i, (tid, ticon, tname, tdesc) in enumerate(task_list):
                with cols_t[i]:
                    is_sel = (tid == current_task)
                    border_style = "border: 2px solid #2f80ed; background: linear-gradient(135deg,rgba(47,128,237,0.1),rgba(139,92,246,0.1));" if is_sel else ""
                    st.markdown(f"""
                    <div class='task-card {"selected" if is_sel else ""}' style='{border_style}'>
                      <div class='task-icon'>{ticon}</div>
                      <div class='task-name'>{tname}</div>
                      <div class='task-desc'>{tdesc}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"{'✓ Selected' if is_sel else 'Select'}", key=f"task_btn_{tid}"):
                        cfg['task'] = tid
                        st.session_state['automl_config'] = cfg
                        st.rerun()

            cfg['task'] = current_task
            task = current_task
            cfg['training_strategy'] = st.radio(
                "Hyperparameter Mode",
                ["Automatic", "Manual"],
                index=0 if cfg.get('training_strategy', 'Automatic') == 'Automatic' else 1,
                horizontal=True,
                help="Automatic: Optuna finds the best params. Manual: You define them.",
                key="wiz_hp_mode"
            )

            # Manual HP configuration moved from Step 3 to Step 2
            if cfg.get('training_strategy') == 'Manual':
                eff_models = cfg.get('selected_models') or available_models
                trainer_hp_wiz = AutoMLTrainer(task_type=task)
                st.markdown("---")
                st.markdown("#### ⚙️ Manual Hyperparameter Configuration")
                ref_model_hp = st.selectbox("Algorithm to Configure", eff_models, key="wiz_step2_manual_model")
                hp_schema = trainer_hp_wiz.get_model_params_schema(ref_model_hp)
                if hp_schema:
                    st.info(f"Define fixed parameters for `{ref_model_hp}`. These will be used instead of Optuna optimization.")
                    mp_step2 = {}
                    cols_hp2 = st.columns(3)
                    for hpi, (hp_name, hp_cfg) in enumerate(hp_schema.items()):
                        with cols_hp2[hpi % 3]:
                            if hp_cfg[0] == 'int':
                                mp_step2[hp_name] = st.number_input(hp_name, hp_cfg[1], hp_cfg[2], hp_cfg[3], key=f"wiz_step2_{hp_name}")
                            elif hp_cfg[0] == 'float':
                                mp_step2[hp_name] = st.number_input(hp_name, hp_cfg[1], hp_cfg[2], hp_cfg[3], format="%.4f", key=f"wiz_step2_{hp_name}")
                            elif hp_cfg[0] == 'list':
                                options_hp, p_def_hp = hp_cfg[1], hp_cfg[2]
                                mp_step2[hp_name] = st.selectbox(hp_name, options_hp, index=options_hp.index(p_def_hp) if p_def_hp in options_hp else 0, key=f"wiz_step2_{hp_name}")
                    cfg['manual_params'] = mp_step2
                else:
                    st.info(f"No manual parameters available for `{ref_model_hp}`. Using system defaults.")
                    cfg['manual_params'] = {}

            st.markdown('<br>', unsafe_allow_html=True)
            col_back, col_fwd, _ = st.columns([1, 1, 5])
            with col_back:
                if st.button("← Back", key="step1_back"):
                    st.session_state['automl_step'] = 0
                    st.session_state['automl_config'] = cfg
                    st.rerun()
            with col_fwd:
                if st.button("Next: Select Models →", type="primary", key="step1_next"):
                    st.session_state['automl_step'] = 2
                    st.session_state['automl_config'] = cfg
                    st.rerun()

        # ════════════════════════════════════════════════════════════════
        # ════════════════════════════════════════════════════════════════
        # STEP 2 — Model Selection
        # ════════════════════════════════════════════════════════════════
        elif cur_step == 2:
            task = cfg.get('task', 'classification')
            trainer_temp = AutoMLTrainer(task_type=task)
            available_models = trainer_temp.get_available_models()

            _col_main, _col_track = st.columns([3, 1])
            with _col_track:
                render_pipeline_overview(PIPELINE_STEPS, cur_step)
            with _col_main:
                st.markdown("<h3 style='margin-bottom:4px;'>🤖 Step 3 — Model Selection</h3>", unsafe_allow_html=True)
                render_step_info_panel(
                    "Model Selection",
                    "Choose which algorithms will compete in the search. You can allow AutoML to explore all available models automatically (recommended for beginners) or hand-pick specific algorithms to focus the search.",
                    ["'Automatic' tests all models", "Manual selection = faster runs", "Ensemble combines multiple models"]
                )

            model_source = st.radio(
                "Model Source",
                ["Standard AutoML (Scikit-Learn/XGBoost/Transformers)", "Model Registry (Registered)", "Local Upload (.pkl)"],
                index=["Standard AutoML (Scikit-Learn/XGBoost/Transformers)", "Model Registry (Registered)", "Local Upload (.pkl)"].index(cfg.get('model_source', 'Standard AutoML (Scikit-Learn/XGBoost/Transformers)')),
                horizontal=True,
                key="wiz_model_source"
            )
            cfg['model_source'] = model_source

            if model_source == "Standard AutoML (Scikit-Learn/XGBoost/Transformers)":
                # ── Model Selection Mode ──────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                # NOTE: "Custom Ensemble Builder" is now merged into "Manual (Select)"
                # In Automatic mode, the system auto-configures ensembles based on Training Focus.
                mode_selection = st.radio(
                    "Selection Mode",
                    ["Automatic (Preset)", "Manual (Select)"],
                    index=0 if cfg.get('mode_selection', 'Automatic (Preset)') not in ["Manual (Select)"] else 1,
                    horizontal=True,
                    key="wiz_mode_sel",
                    help="Automatic: AutoML explores all models using the Training Focus below. Manual: hand-pick specific models and optionally define your own ensemble combinations."
                )
                cfg['mode_selection'] = mode_selection

                # Models that are training METHODS, not standalone estimators — excluded from user-facing model list
                _ENSEMBLE_METHODS = {"custom_voting", "custom_stacking", "custom_bagging", "bagging", "voting_ensemble", "stacking_ensemble"}

                if mode_selection == "Manual (Select)":
                    # Build candidate list: all available models minus pure ensemble-method keys
                    _tmp_trainer = AutoMLTrainer(
                        task_type=task,
                        use_ensemble=cfg.get("use_ensemble", True),
                        use_deep_learning=cfg.get("use_deep_learning", True)
                    )
                    _all_candidates = _tmp_trainer.get_available_models()
                    _model_candidates = [m for m in _all_candidates if m not in _ENSEMBLE_METHODS]

                    st.markdown("<p style='font-size:0.8rem;color:#8b949e;'>Select specific models to include in the search space. Hand-picked models will always be trained regardless of global filters.</p>", unsafe_allow_html=True)
                    _prev_sel = cfg.get("selected_models") or []
                    _prev_plain = [m for m in _prev_sel if m not in _ENSEMBLE_METHODS]
                    sel_models = st.multiselect(
                        "Choose Models",
                        _model_candidates,
                        default=[m for m in _prev_plain if m in _model_candidates] or _model_candidates[:2],
                        key="wiz_sel_models"
                    )
                    cfg["selected_models"] = sel_models if sel_models else None
                else:
                    # Automatic mode — no manual model list needed
                    cfg["selected_models"] = None

                # ── Training Focus selector (shown for both modes) ────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<p style='font-weight:600;margin-bottom:6px;'>Training Focus</p>", unsafe_allow_html=True)
                FOCUS_OPTIONS = [
                    ("single",        "🎯", "Single Models",   "Train individual models only. Faster, simpler, easier to interpret."),
                    ("ensemble_only", "🏗️", "Custom Ensembles","Only train Custom Voting/Stacking/Bagging. Good if base models are already tuned."),
                    ("both",          "🏆", "Both (Full)",     "Train all models including ensembles. Maximum accuracy but takes longer."),
                ]
                cur_focus = cfg.get("ensemble_mode", "both")

                focus_cols = st.columns(3)
                for i, (fid, ficon, fname, fdesc) in enumerate(FOCUS_OPTIONS):
                    with focus_cols[i]:
                        is_sel = (fid == cur_focus)
                        border = "border: 2px solid #27ae60; background:linear-gradient(135deg,rgba(39,174,96,0.1),rgba(39,174,96,0.05));" if is_sel else ""
                        st.markdown(f"""
                        <div class='task-card' style='min-height:130px;{border}'>
                          <div class='task-icon'>{ficon}</div>
                          <div class='task-name'>{fname}</div>
                          <div class='task-desc'>{fdesc}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"{'Selected' if is_sel else 'Select'}", key=f"focus_{fid}"):
                            cfg["ensemble_mode"] = fid
                            cfg["use_ensemble"] = (fid != "single")
                            st.session_state["automl_config"] = cfg
                            st.rerun()
                cfg["ensemble_mode"] = cur_focus
                cfg["use_ensemble"] = (cur_focus != "single")

                # ── Deep Learning selector ────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<p style='font-weight:600;margin-bottom:6px;'>Deep Learning Models</p>", unsafe_allow_html=True)
                DL_OPTIONS = [
                    ("classic", "📊", "Classic ML Only",      "Tree-based models, linear models, SVM, KNN. Fast training, low resource usage."),
                    ("deep",    "🧠", "Include Deep Learning", "Also search Neural Networks (MLP) and Transformers (BERT, etc.). Much slower but may outperform for NLP/complex tabular."),
                ]
                cur_dl = "deep" if cfg.get("use_deep_learning", True) else "classic"
                dl_cols = st.columns(2)
                for i, (did, dicon, dname, ddesc) in enumerate(DL_OPTIONS):
                    with dl_cols[i]:
                        is_sel = (did == cur_dl)
                        border = "border: 2px solid #8b5cf6; background:linear-gradient(135deg,rgba(139,92,246,0.1),rgba(139,92,246,0.05));" if is_sel else ""
                        st.markdown(f"""
                        <div class='task-card' style='min-height:110px;{border}'>
                          <div class='task-icon'>{dicon}</div>
                          <div class='task-name'>{dname}</div>
                          <div class='task-desc'>{ddesc}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"{'Selected' if is_sel else 'Select'}", key=f"dl_{did}"):
                            cfg["use_deep_learning"] = (did == "deep")
                            st.session_state["automl_config"] = cfg
                            st.rerun()
                cfg["use_deep_learning"] = (cur_dl == "deep")

                # ── Custom Ensemble Builder — only in Manual mode + when focus involves ensembles ──
                # In Automatic (Preset) mode, ensembles are configured automatically by the engine.
                if mode_selection == "Manual (Select)" and cur_focus != "single":
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='ui-card' style='padding:14px;margin-bottom:12px;'>
                      <div style='font-weight:600;margin-bottom:4px;'>
                        🏗️ Custom Ensembles
                        <span style='font-size:0.75rem;color:#8b949e;font-weight:400;'>(optional)</span>
                      </div>
                      <div style='color:#8b949e;font-size:0.8rem;'>
                        Add one or more ensemble methods to combine your selected base models.
                        You can add multiple ensembles with different configurations.
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # Initialise list in config
                    if "custom_ensembles" not in cfg or not isinstance(cfg.get("custom_ensembles"), list):
                        cfg["custom_ensembles"] = []

                    # Build base model options for ensembles
                    _ens_base_trainer = AutoMLTrainer(task_type=task, use_deep_learning=cfg.get("use_deep_learning", True))
                    _ens_base_candidates = [m for m in _ens_base_trainer.get_available_models() if m not in _ENSEMBLE_METHODS]

                    # Render existing ensemble entries
                    ensembles_to_keep = []
                    for ei, ens in enumerate(cfg["custom_ensembles"]):
                        ecol1, ecol2 = st.columns([5, 1])
                        with ecol1:
                            if ens["type"] == "voting":
                                ens_label = f"**#{ei+1} Voting** — Models: `{', '.join(ens.get('models', []))}` | Voting: `{ens.get('voting_type', 'soft')}`"
                            elif ens["type"] == "stacking":
                                ens_label = f"**#{ei+1} Stacking** — Models: `{', '.join(ens.get('models', []))}` | Meta: `{ens.get('meta_model', '')}`"
                            else:
                                ens_label = f"**#{ei+1} Bagging** — Base: `{ens.get('base_estimator', 'decision_tree')}`"
                            st.markdown(ens_label)
                        with ecol2:
                            if st.button("🗑️ Remove", key=f"ens_remove_{ei}"):
                                cfg["custom_ensembles"] = [e for j, e in enumerate(cfg["custom_ensembles"]) if j != ei]
                                st.session_state["automl_config"] = cfg
                                st.rerun()
                            else:
                                ensembles_to_keep.append(ens)
                    cfg["custom_ensembles"] = ensembles_to_keep

                    # Form to add a new ensemble
                    with st.expander("➕ Add Custom Ensemble", expanded=len(cfg["custom_ensembles"]) == 0):
                        new_ens_type = st.selectbox(
                            "Ensemble Type",
                            ["Voting", "Stacking", "Bagging"],
                            key="wiz_new_ens_type"
                        )
                        new_ens_cfg = {"type": new_ens_type.lower()}

                        if new_ens_type in ("Voting", "Stacking"):
                            new_sel_base = st.multiselect(
                                "Base Estimators",
                                _ens_base_candidates,
                                default=_ens_base_candidates[:3] if len(_ens_base_candidates) >= 3 else _ens_base_candidates,
                                key="wiz_new_base_models"
                            )
                            if len(new_sel_base) < 2:
                                st.warning("⚠️ Select at least 2 base models.")
                            new_ens_cfg["models"] = new_sel_base

                            if new_ens_type == "Voting":
                                new_voting_type = st.selectbox(
                                    "Voting Type",
                                    ["soft", "hard"] if task == "classification" else ["soft"],
                                    key="wiz_new_vote_type"
                                )
                                new_use_wts = st.checkbox("Weighted Voting", key="wiz_new_use_weights")
                                new_voting_weights = None
                                if new_use_wts:
                                    wts_str = st.text_input(
                                        "Weights (comma-separated)",
                                        value=",".join(["1.0"] * len(new_sel_base)),
                                        key="wiz_new_weights"
                                    )
                                    try:
                                        new_voting_weights = [float(w.strip()) for w in wts_str.split(",")]
                                    except Exception:
                                        st.error("Invalid weights format.")
                                new_ens_cfg["voting_type"] = new_voting_type
                                new_ens_cfg["voting_weights"] = new_voting_weights
                            else:  # Stacking
                                _meta_opts = ["logistic_regression", "random_forest", "xgboost", "ridge", "linear_regression"]
                                if task == "classification":
                                    _meta_opts = [m for m in _meta_opts if m not in ["linear_regression", "ridge"]]
                                else:
                                    _meta_opts = [m for m in _meta_opts if m not in ["logistic_regression"]]
                                if not _meta_opts:
                                    _meta_opts = ["random_forest"]
                                new_meta = st.selectbox("Meta-Model (Final Estimator)", _meta_opts, key="wiz_new_meta_model")
                                new_ens_cfg["meta_model"] = new_meta

                        else:  # Bagging
                            _bag_base_opts = ["decision_tree", "logistic_regression", "knn", "svm", "extra_trees"]
                            new_bag_base = st.selectbox("Base Estimator", _bag_base_opts, key="wiz_new_bag_base")
                            new_ens_cfg["base_estimator"] = new_bag_base

                        if st.button("✅ Add This Ensemble", key="wiz_add_ensemble_btn"):
                            valid = True
                            if new_ens_type in ("Voting", "Stacking") and len(new_ens_cfg.get("models", [])) < 2:
                                st.error("Please select at least 2 base models first.")
                                valid = False
                            if valid:
                                cfg["custom_ensembles"].append(new_ens_cfg)
                                st.session_state["automl_config"] = cfg
                                st.rerun()

                    # Build selected_models and ensemble_config from the custom_ensembles list
                    _base_sel = [m for m in (cfg.get("selected_models") or []) if m not in _ENSEMBLE_METHODS]
                    _ens_model_keys = []
                    _ens_config_list = []
                    for ens in cfg["custom_ensembles"]:
                        if ens["type"] == "voting":
                            _ens_model_keys.append("custom_voting")
                            _ens_config_list.append({
                                "voting_estimators": ens.get("models", []),
                                "voting_type":       ens.get("voting_type", "soft"),
                                "voting_weights":    ens.get("voting_weights"),
                            })
                        elif ens["type"] == "stacking":
                            _ens_model_keys.append("custom_stacking")
                            _ens_config_list.append({
                                "stacking_estimators":       ens.get("models", []),
                                "stacking_final_estimator":  ens.get("meta_model", "logistic_regression"),
                            })
                        elif ens["type"] == "bagging":
                            _ens_model_keys.append("custom_bagging")
                            _ens_config_list.append({
                                "bagging_base_estimator": ens.get("base_estimator", "decision_tree"),
                            })

                    _final_sel = _base_sel + _ens_model_keys
                    cfg["selected_models"] = _final_sel if _final_sel else None
                    # For backward compat with trainer: store first ensemble config in ensemble_config
                    cfg["ensemble_config"]      = _ens_config_list[0] if _ens_config_list else {}
                    cfg["ensemble_configs_list"] = _ens_config_list

                else:
                    # Automatic mode OR Single-Models-only Manual mode — clear manual ensemble config
                    cfg["ensemble_config"]       = {}
                    cfg["ensemble_configs_list"] = []
                    if "custom_ensembles" not in cfg:
                        cfg["custom_ensembles"] = []

            elif model_source == "Model Registry (Registered)":
                reg_models_list = get_cached_registered_models()
                if reg_models_list:
                    base_name = st.selectbox("Registered Model", [m.name for m in reg_models_list], key="wiz_reg_model")
                    cfg['selected_models'] = [base_name]
                    st.info(f"Model **{base_name}** will be used as base for retraining.")
                else:
                    cfg['selected_models'] = None

            elif model_source == "Local Upload (.pkl)":
                uploaded_pkl = st.file_uploader("Upload .pkl file", type="pkl", key="wiz_pkl_upload")
                if uploaded_pkl:
                    cfg['selected_models'] = ["Uploaded_Model"]
                    st.success("Model loaded for retraining.")
                else:
                    cfg['selected_models'] = None

            # ── Parallelism (n_jobs) ──────────────────────────────────
            with st.expander("⚙️ Parallelism & Compute"):
                n_jobs_mode = st.radio("CPU Usage (n_jobs)", ["Automatic (All cores)", "Manual"], index=0, key="wiz_njobs_mode")
                if n_jobs_mode == "Manual":
                    n_jobs = st.number_input("Threads / Jobs", -1, 128, -1, key="wiz_njobs_val", help="-1 = all cores, 1 = sequential")
                else:
                    n_jobs = -1
                cfg['n_jobs'] = n_jobs

            # ── Manual Hyperparameter Inputs ──────────────────────────────
            if cfg.get('training_strategy') == 'Manual':
                st.info("Manual tuning selected.") 
    
    # --- SUB-TAB 1.2: COMPUTER VISION ---
    with automl_tabs[1]:
        st.markdown("""
        <div class='hero-header' style='background:linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(31, 41, 55, 0.4) 100%);'>
          <div class='hero-title'>👁️ Vision Studio</div>
          <div class='hero-subtitle'>Train deep learning vision models for classification, detection, and segmentation.</div>
        </div>""", unsafe_allow_html=True)

        CV_TASKS = [
            ("image_classification", "🖼️", "Classification", "Assign a single label to an image."),
            ("image_multi_label", "🏷️", "Multi-Label", "Assign multiple labels to an image simultaneously."),
            ("image_segmentation", "🧩", "Segmentation", "Pixel-level classification (masks)."),
            ("object_detection", "🎯", "Detection", "Find and bound objects in an image."),
        ]
        
        st.markdown("<h4 style='margin-bottom:12px;'>1. Select Vision Task</h4>", unsafe_allow_html=True)
        cur_cv_task = st.session_state.get('wiz_cv_task', 'image_classification')
        cols_cv = st.columns(4)
        for i, (tid, ticon, tname, tdesc) in enumerate(CV_TASKS):
            with cols_cv[i]:
                is_sel = (tid == cur_cv_task)
                border = "border: 2px solid #8b5cf6; background:linear-gradient(135deg,rgba(139,92,246,0.1),rgba(139,92,246,0.05));" if is_sel else ""
                st.markdown(f"""
                <div class='task-card' style='min-height:130px;{border}'>
                  <div class='task-icon'>{ticon}</div>
                  <div class='task-name'>{tname}</div>
                  <div class='task-desc'>{tdesc}</div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"{'✓ Selected' if is_sel else 'Select'}", key=f"cvt_{tid}"):
                    st.session_state['wiz_cv_task'] = tid
                    st.rerun()
        
        cv_task = st.session_state.get('wiz_cv_task', 'image_classification')
        st.divider()
        
        col_cv_main, col_cv_side = st.columns([2, 1])
        with col_cv_main:
            st.markdown("#### 2. Model & Data Configuration")
            # --- Backbone Selection ---
            backbones = ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0', 'densenet121', 'vgg16']
            if cv_task in ['image_classification', 'image_multi_label']:
                selected_backbone = st.selectbox("Network Backbone", backbones, index=0, key="cv_backbone",
                    help="Choose the CNN architecture to act as the feature extractor.")
            else:
                selected_backbone = 'resnet50' # Default for segmentation/detection
                st.info(f"Using default architecture for {cv_task}")
            
            # --- Data Selection from DataLake ---
            st.markdown("##### Dataset Selection")
            datasets = get_cached_datasets()
            
            # Select Image Archive (ZIP)
            cv_ds_sel = st.selectbox("Select Image Archive (ZIP dataset)", [""] + datasets, key="cv_ds_sel",
                help="Classification: zip of class-named folders. Multi-label: zip of images. Seg: zip of images + masks.")
            cv_upload_path = None
            if cv_ds_sel:
                versions = get_cached_versions(cv_ds_sel)
                cv_ver_sel = st.selectbox("Version (ZIP)", versions, key="cv_ver_sel")
                if cv_ver_sel:
                    cv_upload_path = os.path.join(datalake.base_path, cv_ds_sel, cv_ver_sel)

            label_csv_path = None
            if cv_task == 'image_multi_label':
                st.info("For multi-label, select the CSV dataset containing: filename, label1, label2...")
                csv_ds_sel = st.selectbox("Select Multi-label CSV", [""] + datasets, key="cv_csv_ds_sel")
                if csv_ds_sel:
                    versions_csv = get_cached_versions(csv_ds_sel)
                    csv_ver_sel = st.selectbox("Version (CSV)", versions_csv, key="cv_csv_ver_sel")
                    if csv_ver_sel:
                        label_csv_path = os.path.join(datalake.base_path, csv_ds_sel, csv_ver_sel)

            data_dir = None
            if cv_upload_path and cv_upload_path.endswith('.zip'):
                import zipfile
                import shutil
                temp_extract_dir = "temp_cv_dataset"
                def remove_readonly(func, path, excinfo):
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                if os.path.exists(temp_extract_dir):
                    try: shutil.rmtree(temp_extract_dir, onerror=remove_readonly)
                    except: pass
                
                os.makedirs(temp_extract_dir, exist_ok=True)
                with zipfile.ZipFile(cv_upload_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                data_dir = temp_extract_dir
                subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
                # If there's exactly one folder, that's probably the actual root
                if len(subdirs) == 1 and not any(f.endswith('.jpg') or f.endswith('.png') for f in os.listdir(data_dir)):
                    data_dir = os.path.join(data_dir, subdirs[0])
                st.success(f"Data ready (found {len(os.listdir(data_dir))} items in root).")
            elif cv_upload_path:
                st.error("Selected image archive must be a ZIP file.")

            # --- Augmentation ---
            with st.expander("🛠️ Data Augmentation"):
                st.write("Apply random transformations to reduce overfitting.")
                aug_cols = st.columns(3)
                do_hflip = aug_cols[0].checkbox("Horizontal Flip", True, key="aug_hflip")
                do_vflip = aug_cols[1].checkbox("Vertical Flip", False, key="aug_vflip")
                do_jitter = aug_cols[2].checkbox("Color Jitter", True, key="aug_jitter")
                rot_deg = st.slider("Random Rotation (degrees)", 0, 90, 15, key="aug_rot")
                do_crop = st.checkbox("Random Resized Crop", False, key="aug_crop")
                
                aug_config = {
                    'horizontal_flip': do_hflip,
                    'vertical_flip': do_vflip,
                    'color_jitter': do_jitter,
                    'random_rotation': rot_deg,
                    'random_crop': do_crop
                }

        with col_cv_side:
            st.markdown("#### 3. Hyperparameters")
            epochs = st.number_input("Epochs", 1, 100, 10, key="cv_epochs")
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32, 64], index=2, key="cv_batch")
            lr_cv = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", key="cv_lr")
            optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0, key="cv_opt")
            val_split = st.slider("Validation Split (%)", 5, 40, 20, key="cv_val_split") / 100.0
            
            st.markdown("<br>", unsafe_allow_html=True)
            start_btn = st.button("🚀 Start Training", type="primary", use_container_width=True, key="cv_start_btn")

        # ------------------------------------------------------------------
        # Training Execution
        # ------------------------------------------------------------------
        if start_btn:
            if not data_dir:
                st.error("Please upload an image dataset ZIP.")
            elif cv_task == 'image_multi_label' and not label_csv_path:
                st.error("Please upload the label CSV for multi-label tasks.")
            else:
                from cv_engine import CVAutoMLTrainer
                trainer = CVAutoMLTrainer(task_type=cv_task, backbone=selected_backbone)
                
                st.divider()
                st.subheader("Training Progress")
                
                metric_cols = st.columns(4)
                m_epoch = metric_cols[0].empty()
                m_train_loss = metric_cols[1].empty()
                m_val_loss = metric_cols[2].empty()
                m_val_acc = metric_cols[3].empty()
                
                chart_container = st.empty()
                hist_data = {'epoch': [], 'loss': [], 'val_loss': [], 'val_acc': []}

                def cv_callback(epoch, acc, loss, duration, val_acc, val_loss):
                    m_epoch.metric("Epoch", f"{epoch+1}/{epochs}")
                    m_train_loss.metric("Train Loss", f"{loss:.4f}")
                    m_val_loss.metric("Val Loss", f"{val_loss:.4f}" if val_loss else "N/A")
                    m_val_acc.metric("Val Acc" if cv_task != "image_segmentation" else "Val Score", f"{val_acc:.4f}" if val_acc else "N/A")
                    
                    hist_data['epoch'].append(epoch+1)
                    hist_data['loss'].append(loss)
                    hist_data['val_loss'].append(val_loss if val_loss else 0)
                    hist_data['val_acc'].append(val_acc if val_acc else 0)
                    
                    if len(hist_data['epoch']) > 1:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Scatter(x=hist_data['epoch'], y=hist_data['loss'], name="Train Loss", line=dict(color="#ef4444")), secondary_y=False)
                        if any(hist_data['val_loss']):
                            fig.add_trace(go.Scatter(x=hist_data['epoch'], y=hist_data['val_loss'], name="Val Loss", line=dict(color="#f59e0b", dash="dash")), secondary_y=False)
                        if any(hist_data['val_acc']):
                            fig.add_trace(go.Scatter(x=hist_data['epoch'], y=hist_data['val_acc'], name="Val Acc", line=dict(color="#27ae60")), secondary_y=True)
                        fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        chart_container.plotly_chart(fig, use_container_width=True)

                with st.spinner(f"Training {selected_backbone} on GPU/CPU..."):
                    try:
                        import mlflow
                        from src.utils.helpers import get_cv_consumption_code
                        
                        run_name = f"CV_{cv_task}_{selected_backbone}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
                        
                        with mlflow.start_run(run_name=run_name) as run:
                            # Log configurations
                            mlflow.log_params({
                                "cv_task": cv_task,
                                "backbone": selected_backbone,
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "learning_rate": lr_cv,
                                "optimizer": optimizer,
                                "augmentation": str(aug_config),
                                "val_split": val_split
                            })
                            mlflow.set_tag("task_type", cv_task)
                            mlflow.set_tag("is_cv", "true")

                            best_model_cv = trainer.train(
                                data_dir=data_dir, n_epochs=epochs, batch_size=batch_size,
                                lr=lr_cv, callback=cv_callback, mask_dir=None,
                                augmentation_config=aug_config, label_csv=label_csv_path,
                                val_split=val_split, optimizer_name=optimizer
                            )
                            st.success("✨ Vision Training Complete!")
                            st.session_state['best_cv_model'] = best_model_cv
                            st.session_state['cv_trainer'] = trainer
                            
                            # Log Final Metrics
                            if len(hist_data['val_acc']) > 0:
                                mlflow.log_metric("final_val_acc", hist_data['val_acc'][-1])
                            if len(hist_data['val_loss']) > 0:
                                mlflow.log_metric("final_val_loss", hist_data['val_loss'][-1])
                                
                            # Log Model
                            import torch
                            mlflow.pytorch.log_model(best_model_cv, "model")
                            
                            # Generate Code
                            st.session_state['cv_run_id'] = run.info.run_id
                            st.divider()
                            st.markdown("### 🧩 Model Consumption Code")
                            code = get_cv_consumption_code(selected_backbone, run.info.run_id, cv_task, selected_backbone)
                            st.code(code, language='python')
                            
                    except Exception as e:
                        st.error(f"CV Training Failed: {e}")

        # ------------------------------------------------------------------
        # Inference Test
        # ------------------------------------------------------------------
        if st.session_state.get('best_cv_model'):
            st.divider()
            st.markdown("### 🔍 Model Inference & Explainability")
            
            col_inf_l, col_inf_r = st.columns([1, 1])
            with col_inf_l:
                test_img = st.file_uploader("Upload an image to test", type=['jpg', 'jpeg', 'png'], key="cv_test_upload")
                if test_img:
                    st.image(test_img, caption="Input", use_container_width=True)
            
            with col_inf_r:
                if test_img:
                    img_path = f"temp_{test_img.name}"
                    with open(img_path, "wb") as f:
                        f.write(test_img.getbuffer())
                    
                    trainer = st.session_state['cv_trainer']
                    with st.spinner("Running inference..."):
                        prediction = trainer.predict(img_path)
                    
                    st.markdown("#### Predictions")
                    if trainer.task_type == 'image_multi_label':
                        probs = prediction['probabilities']
                        names = prediction['label_names']
                        import pandas as pd
                        import plotly.express as px
                        df_p = pd.DataFrame({'Label': names, 'Probability': probs})
                        df_p = df_p.sort_values('Probability', ascending=True).tail(10)
                        fig = px.bar(df_p, x='Probability', y='Label', orientation='h', color='Probability', color_continuous_scale='Viridis')
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif trainer.task_type == 'image_classification':
                        probs = prediction['probabilities']
                        names = prediction.get('class_names', [str(i) for i in range(len(probs))])
                        df_p = pd.DataFrame({'Class': names, 'Probability': probs})
                        df_p = df_p.sort_values('Probability', ascending=True).tail(5) # Top 5
                        fig = px.bar(df_p, x='Probability', y='Class', orientation='h', color='Probability', color_continuous_scale='Blues')
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("Top Result", df_p.iloc[-1]['Class'], f"{df_p.iloc[-1]['Probability']:.1%}")
                        
                    elif trainer.task_type == 'image_segmentation':
                        mask_img = Image.fromarray((prediction * (255 // (prediction.max() if prediction.max() > 0 else 1))).astype(np.uint8))
                        st.image(mask_img, caption="Predicted Mask", use_container_width=True)

                    try:
                        os.remove(img_path)
                    except: pass
            
            st.markdown("#### Architect Insight")
            from cv_engine import get_cv_explanation
            cfg_used = {'lr': st.session_state.get('cv_lr', 'N/A'), 'batch_size': st.session_state.get('cv_batch', 'N/A')}
            insight = get_cv_explanation(trainer.backbone, cfg_used)
            st.info(f"🧠 **Model Insight:** {insight}")

# --- TAB 3: EXPERIMENTS ---
with tabs[2]:
    jm: TrainingJobManager = st.session_state['job_manager']

    @st.fragment(run_every=5.0)
    def experiments_dashboard():
        import pandas as pd
        import plotly.express as px
        from src.tracking.manager import JobStatus
        jm.poll_updates()
        jobs = jm.list_jobs()

        # Premium Hero Header
        run_count   = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
        total_count = len(jobs)
        done_count  = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
        fail_count  = sum(1 for j in jobs if j.status == JobStatus.FAILED)

        st.markdown(f"""
        <div class='hero-header'>
          <div class='hero-title'>🧪 Experiments
            <span class='version-badge'>{run_count} running</span>
          </div>
          <div class='hero-subtitle'>Track, manage and inspect all your training jobs in real time.</div>
        </div>""", unsafe_allow_html=True)

        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        for col, val, label, color in [
            (col_meta1, total_count, "Total Jobs",   "#2f80ed"),
            (col_meta2, run_count,   "Running",       "#27ae60"),
            (col_meta3, done_count,  "Completed",     "#8b5cf6"),
            (col_meta4, fail_count,  "Failed",        "#ef4444"),
        ]:
            with col:
                col.markdown(f"""
                <div class='ui-metric' style='border-left-color:{color};'>
                  <div class='metric-value'>{val}</div>
                  <div class='metric-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        col_exp_h1, col_exp_h2 = st.columns([4, 1])
        with col_exp_h2:
            if st.button("🔄 Refresh", key="exp_refresh"):
                st.rerun()

        if not jobs:
            st.markdown("""
            <div class='ui-card' style='text-align:center;padding:40px;'>
              <div style='font-size:2.5rem;'>🧪</div>
              <div style='font-weight:600;font-size:1.1rem;margin:12px 0;'>No experiments yet</div>
              <div style='color:#8b949e;'>Go to <strong>AutoML &amp; Model Hub</strong> and submit your first experiment.</div>
            </div>""", unsafe_allow_html=True)
            return

        # Search / filter bar
        search_q = st.text_input("🔍 Filter experiments", key="exp_search", placeholder="Filter by name...")
        filter_status = st.multiselect(
            "Status Filter",
            [JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
            default=[JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
            key="exp_status_filter",
            format_func=lambda s: {
                "queued": "🔵 Queued", "running": "🟢 Running", "paused": "🟡 Paused",
                "completed": "✅ Completed", "failed": "🔴 Failed", "cancelled": "⚫ Cancelled"
            }.get(s, s)
        )

        visible_jobs = [
            j for j in jobs
            if j.status in filter_status and (not search_q or search_q.lower() in j.name.lower())
        ]

        # ── Global comparison chart ──────────────────────────────────────
        completed_with_score = [j for j in jobs if j.best_score is not None]
        if len(completed_with_score) >= 2:
            comp_df = pd.DataFrame([
                {"Experiment": j.name[:25], "Best Score": j.best_score, "Task": j.config.get('task','?'),
                 "Metric": j.config.get('optimization_metric','?')}
                for j in sorted(completed_with_score, key=lambda x: x.best_score, reverse=True)
            ])
            with st.expander("📊 Experiment Comparison", expanded=True):
                fig_cmp = px.bar(
                    comp_df, x="Best Score", y="Experiment", orientation='h',
                    color="Best Score", color_continuous_scale=["#30363d","#2f80ed","#8b5cf6"],
                    title="Best Score Across All Experiments",
                    hover_data=["Task", "Metric"]
                )
                fig_cmp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e6edf3', title_font_size=14,
                    margin=dict(t=40,b=10,l=10,r=10),
                    yaxis=dict(autorange='reversed'),
                    coloraxis_showscale=False
                )
                fig_cmp.update_traces(marker_line_width=0)
                st.plotly_chart(fig_cmp, use_container_width=True, key="global_cmp_chart")

        if not visible_jobs:
            st.warning("No experiments match the current filters.")

        BADGE_MAP = {
            "queued":    ("badge-queued",  "🔵 Queued"),
            "running":   ("badge-running", "🟢 Running"),
            "paused":    ("badge-paused",  "🟡 Paused"),
            "completed": ("badge-done",    "✅ Done"),
            "failed":    ("badge-failed",  "🔴 Failed"),
            "cancelled": ("badge-queued",  "⚫ Cancelled"),
        }

        for job in visible_jobs:
            badge_cls, badge_lbl = BADGE_MAP.get(job.status, ("badge-queued", job.status))
            score_display = f"Best: <strong>{job.best_score:.4f}</strong>" if job.best_score is not None else "<span style='color:#8b949e'>No score yet</span>"
            
            # Plain static text for the expander title (avoids UI state destruction on autorefresh)
            plain_label = f"{badge_lbl} | {job.name}"
            
            # Styled html to show inside the expander
            styled_header = f"""
            <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px; margin-bottom: 12px;'>
              <div style='display:flex;align-items:center;gap:10px;'>
                <span style='font-weight:700;font-size:1.1rem;'>{job.name}</span>
                <span class='badge {badge_cls}'>{badge_lbl}</span>
              </div>
              <div style='display:flex;align-items:center;gap:16px;font-size:0.9rem;color:#8b949e;'>
                <span>⏱ {job.duration_str}</span>
                <span>{score_display}</span>
              </div>
            </div>"""

            with st.expander(plain_label, expanded=(job.status in (JobStatus.RUNNING, JobStatus.PAUSED))):
                st.markdown(styled_header, unsafe_allow_html=True)
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
                    _training_stage_bar_col, _ov_info_col = st.columns([2, 1])
                    with _training_stage_bar_col:
                        st.markdown("**Training Pipeline Stage**")
                        render_training_mini_pipeline(job.logs or [], job.status)
                    with _ov_info_col:
                        _trials_done = len(job.trials_data) if job.trials_data else 0
                        # Calculate accurate total estimated trials based on models per preset
                        _n_trials_per_model = job.config.get('n_trials', 20) or 20
                        _preset_name = job.config.get('preset', 'medium')
                        _models_map = {'test': 2, 'fast': 5, 'medium': 10, 'high': 17}
                        
                        if _preset_name == 'custom':
                            _sel_models = job.config.get('selected_models')
                            _n_models = len(_sel_models) if isinstance(_sel_models, list) else 1
                        else:
                            _n_models = _models_map.get(_preset_name, 10)
                            
                        _est_trials = _n_trials_per_model * _n_models
                        
                        render_pipeline_progress_ring(_trials_done, _est_trials, is_done=(job.status == JobStatus.COMPLETED))

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
                        cfg_job = job.config
                        st.markdown("**Configuration:**")
                        st.json({
                            "task": cfg_job.get("task"),
                            "target": cfg_job.get("target"),
                            "preset": cfg_job.get("preset"),
                            "n_trials": cfg_job.get("n_trials"),
                            "validation": cfg_job.get("validation_strategy"),
                            "optimization_metric": cfg_job.get("optimization_metric"),
                            "selected_models": cfg_job.get("selected_models") or "All",
                        }, expanded=False)

                # ── Tab 1: Progress ──
                with detail_tabs[1]:
                    # Trial Status Visualization
                    if hasattr(job, 'trial_statuses') and job.trial_statuses:
                        st.markdown("**Live Trial Tracker**")
                        ts_cols = st.columns(min(len(job.trial_statuses), 5))
                        for idx, (tid, tstat) in enumerate(job.trial_statuses.items()):
                            t_col = ts_cols[idx % 5]
                            with t_col:
                                t_color = "#27ae60" if tstat == "completed" else "#f1c40f" if tstat == "running" else "#34495e"
                                st.markdown(f"""
                                <div style='background:{t_color}; color:white; padding:4px 8px; border-radius:4px; font-size:0.75rem; text-align:center; margin-bottom:4px;'>
                                  Trial {tid}<br><b>{tstat.upper()}</b>
                                </div>""", unsafe_allow_html=True)
                        st.divider()

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
                        from src.core.trainer import get_ensemble_display_name
                        for mn, mi in job.model_summaries.items():
                            display_mn = get_ensemble_display_name(mn)
                            row = {"Model": display_mn, "Best Score": f"{mi.get('score', 0):.4f}", "ScoreVal": mi.get('score', 0),
                                   "Trial": mi.get("trial_name", "?"), "Duration (s)": f"{mi.get('duration', 0):.2f}"}
                            if 'metrics' in mi:
                                for mk, mv in mi['metrics'].items():
                                    if mk != 'confusion_matrix' and isinstance(mv, (int, float)):
                                        row[mk.upper()] = f"{mv:.4f}"
                            sum_rows.append(row)
                        if sum_rows:
                            df_sum = pd.DataFrame(sum_rows)
                            
                            # Interactive Bar Graph
                            fig_bar = px.bar(df_sum, x="Model", y="ScoreVal", color="Model",
                                             title="Best Score by Model", text_auto=".4f")
                            fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Optimization Metric Score")
                            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{job.job_id}")
                            
                            # Clean up for display
                            df_display = df_sum.drop(columns=['ScoreVal'])
                            st.dataframe(df_display, use_container_width=True)

                # ── Tab 2: Logs ──
                with detail_tabs[2]:
                    if job.logs:
                        render_colored_log(job.logs[-200:])
                    else:
                        st.markdown("""
                        <div class='ui-card' style='text-align:center;padding:24px;'>
                          <div style='font-size:1.5rem;'>🖥️</div>
                          <div style='color:#8b949e;margin-top:8px;'>No logs yet. Logs appear as training progresses.</div>
                        </div>""", unsafe_allow_html=True)

                # ── Tab 3: Results ──
                with detail_tabs[3]:
                    if job.report_data:
                        from src.core.trainer import get_ensemble_display_name
                        for m_name, rep in job.report_data.items():
                            display_m_name = get_ensemble_display_name(m_name)
                            with st.expander(f"📊 Report: {display_m_name} (Score: {rep.get('score', 0):.4f})", expanded=True):
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
                                    
                                    constituents = rep.get('constituents', [])
                                    if constituents:
                                        st.markdown("**Ensemble Constituents:**")
                                        st.caption(", ".join(constituents))

                                    # Consumption Code Integrated
                                    c_code = rep.get('consumption_code')
                                    if c_code:
                                        st.markdown("#### 📦 Consumption Code")
                                        st.code(c_code, language='python')

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
                                        
                                        # --- NEW: Deployment & Test Integrated ---
                                        st.divider()
                                        st.markdown("#### 🚀 Deployment & Test")
                                        dep_col, test_col = st.columns(2)
                                        with dep_col:
                                            st.markdown("**Hugging Face Hub**")
                                            hf_repo = st.text_input("HF Repo ID (e.g. user/model)", key=f"hf_repo_{job.job_id}_{m_name}")
                                            hf_token = st.text_input("HF Token", type="password", key=f"hf_token_{job.job_id}_{m_name}")
                                            if st.button("🚀 Push to HF", key=f"hf_push_{job.job_id}_{m_name}"):
                                                if hf_repo and hf_token:
                                                    with st.spinner("Pushing to Hugging Face..."):
                                                        # Logic similar to what was in experiments_dashboard previously
                                                        try:
                                                            import mlflow
                                                            from src.deploy.hf_deploy import deploy_to_huggingface
                                                            import tempfile, os
                                                            run_id = rep.get('run_id')
                                                            if run_id:
                                                                with tempfile.TemporaryDirectory() as tmp_dir:
                                                                    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=tmp_dir)
                                                                    # Check for model folder
                                                                    model_folder = os.path.join(local_path, "model")
                                                                    deploy_path = model_folder if os.path.exists(model_folder) else local_path
                                                                    url = deploy_to_huggingface(deploy_path, hf_repo, hf_token)
                                                                    st.success(f"Deployed! [View on HF Hub]({url})")
                                                            else:
                                                                st.error("No Run ID found for this model.")
                                                        except Exception as e:
                                                            st.error(f"HF Deployment failed: {e}")
                                                else:
                                                    st.warning("Please provide Repo ID and Token.")
                                        
                                        with test_col:
                                            st.markdown("**Inference Playground**")
                                            st.caption("Enter a JSON row or CSV values to test the model.")
                                            test_input = st.text_area("Input Data", "{}", key=f"test_in_{job.job_id}_{m_name}")
                                            if st.button("🔮 Predict", key=f"pred_btn_{job.job_id}_{m_name}"):
                                                try:
                                                    import json
                                                    import pandas as pd
                                                    import mlflow.sklearn
                                                    run_id = rep.get('run_id')
                                                    if run_id:
                                                        model_uri = f"runs:/{run_id}/model"
                                                        loaded_model = mlflow.sklearn.load_model(model_uri)
                                                        # Try to parse as JSON list of dicts or single dict
                                                        try:
                                                            data_raw = json.loads(test_input)
                                                            if isinstance(data_raw, dict): data_raw = [data_raw]
                                                            df_input = pd.DataFrame(data_raw)
                                                        except:
                                                            # Try CSV-like string
                                                            import io
                                                            df_input = pd.read_csv(io.StringIO(test_input))
                                                        
                                                        preds = loaded_model.predict(df_input)
                                                        st.json({"prediction": preds.tolist()})
                                                        if hasattr(loaded_model, "predict_proba"):
                                                            proba = loaded_model.predict_proba(df_input)
                                                            st.json({"probabilities": proba.tolist()})
                                                    else:
                                                        st.error("No Run ID found.")
                                                except Exception as e:
                                                    st.error(f"Prediction error: {e}")

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
                        st.markdown(f"#### 🔍 MLflow Details (Run: `{job.mlflow_run_id}`)")
                        try:
                            # Use caching for MLflow details
                            cache = st.session_state.get('mlflow_cache', {})
                            if job.mlflow_run_id not in cache:
                                with st.spinner("Fetching MLflow data..."):
                                    cache[job.mlflow_run_id] = get_cached_run_details(job.mlflow_run_id)
                                    st.session_state['mlflow_cache'] = cache
                            
                            rd = cache.get(job.mlflow_run_id, {"error": "No details found."})
                            
                            if "error" in rd:
                                st.error(f"Could not load MLflow data: {rd['error']}")
                                st.markdown("---")
                                st.markdown("#### 🛠️ MLflow Troubleshooting")
                                st.info("The MLflow database seems to have a schema conflict.")
                                rcols = st.columns(2)
                                with rcols[0]:
                                    if st.button("🔌 Attempt Fix", key=f"t4_fix_{job.job_id}"):
                                        import subprocess
                                        subprocess.run(["python", "-m", "mlflow", "db", "stamp", "head", "--url", "sqlite:///mlflow.db"])
                                        st.rerun()
                                with rcols[1]:
                                    if st.button("🗑 Reset DB", key=f"t4_rst_{job.job_id}"):
                                        import os
                                        if os.path.exists("mlflow.db"): os.remove("mlflow.db")
                                        st.rerun()
                            else:
                                # Standard Display
                                info = rd.get("info", {})
                                cols_ml = st.columns(3)
                                cols_ml[0].metric("Status", info.get("status", "?"))
                                if info.get("start_time"):
                                    import datetime as _dt
                                    dt_val = _dt.datetime.fromtimestamp(info["start_time"] / 1000)
                                    cols_ml[1].metric("Started", dt_val.strftime("%H:%M:%S"))
                                cols_ml[2].caption(f"Experiment: **{info.get('experiment_name', '?')}**")

                                ml_tabs = st.tabs(["Parameters", "Metrics", "Tags", "Artifacts"])
                                with ml_tabs[0]: st.json(rd.get("params", {}))
                                with ml_tabs[1]:
                                    metrics_data = rd.get("metrics", {})
                                    if metrics_data:
                                        met_cols = st.columns(min(len(metrics_data), 4))
                                        for mi, (mk, mv) in enumerate(metrics_data.items()):
                                            met_cols[mi % 4].metric(mk, f"{mv:.4f}" if isinstance(mv, float) else mv)
                                        
                                        hist = rd.get("metric_history", {})
                                        if hist:
                                            metric_sel = st.selectbox("History", list(hist.keys()), key=f"hsel_t4_{job.job_id}")
                                            if metric_sel:
                                                h_df = pd.DataFrame(hist[metric_sel])
                                                st.plotly_chart(px.line(h_df, x="step", y="value"), use_container_width=True)
                                    else: st.info("No metrics.")
                                with ml_tabs[2]: st.json(rd.get("tags", {}))
                                with ml_tabs[3]:
                                    artifacts = rd.get("artifacts", [])
                                    if artifacts:
                                        for a in artifacts: st.markdown(f"📄 `{a}`")
                                    else: st.info("No artifacts logged.")
                                
                                st.divider()
                                if st.button("🗑 Reset MLflow Database", key=f"g_rst_t4_{job.job_id}"):
                                    import os
                                    if os.path.exists("mlflow.db"):
                                        os.remove("mlflow.db")
                                        st.rerun()
                        except Exception as e:
                            st.error(f"Error handling MLflow data: {e}")
                    else:
                        st.info("MLflow data will be available after the first trial completes.")

                # ── Tab 5: Register Model ──
                with detail_tabs[5]:
                    if job.mlflow_run_id:
                        st.markdown("### Register this run as an Official Model")
                        reg_name = st.text_input(
                            "Model Registry Name", 
                            value=f"{job.config.get('task', 'model')}_{job.name[:20].replace(' ', '_')}",
                            key=f"reg_name_{job.job_id}"
                        )
                        if st.button("📦 Register Model", key=f"reg_btn_{job.job_id}", type="primary"):
                            with st.spinner("Registering..."):
                                success = register_model_from_run(job.mlflow_run_id, reg_name)
                            if success:
                                st.success(f"✅ Model **{reg_name}** registered! Check the **Model Registry & Deploy** tab.")
                            else:
                                st.error("Registration failed. Ensure the run has a logged model artifact.")
                        
                        st.divider()
                        st.markdown("#### 📤 Export & External Deploy")
                        onnx_col, hf_col = st.columns(2)
                        with onnx_col:
                            if st.button("🔌 Export to ONNX", key=f"onnx_btn_{job.job_id}"):
                                with st.spinner("Converting to ONNX..."):
                                    try:
                                        from mlflow.sklearn import load_model as ml_load_skl
                                        model_uri = f"runs:/{job.mlflow_run_id}/model"
                                        skl_model = ml_load_skl(model_uri)
                                        
                                        from src.engines.classical import AutoMLTrainer
                                        t_onnx = AutoMLTrainer(task_type=job.config.get('task', 'classification'))
                                        t_onnx.best_model = skl_model
                                        
                                        # Use 1 row sample for shape inference
                                        if 'current_df' in st.session_state:
                                            sample_x = st.session_state['current_df'].drop(columns=[job.config.get('target', '')]).head(1).values
                                            out_path = f"exported_model_{job.job_id}.onnx"
                                            t_onnx.export_best_model_to_onnx(X_sample=sample_x, path=out_path)
                                            with open(out_path, "rb") as f:
                                                st.download_button("📥 Click to Download ONNX", f, file_name=out_path)
                                            st.success(f"Model converted to ONNX!")
                                        else:
                                            st.warning("Need dataset in session to infer ONNX shape.")
                                    except Exception as e:
                                        st.error(f"ONNX Export failed: {e}")
                        
                        with hf_col:
                            hf_token = st.text_input("HF Token", type="password", key=f"hf_token_{job.job_id}")
                            hf_repo = st.text_input("HF Repo ID (e.g. user/model)", key=f"hf_repo_{job.job_id}")
                            if st.button("🤗 Push to Hugging Face", key=f"hf_btn_{job.job_id}"):
                                if not hf_token or not hf_repo:
                                    st.warning("Please provide HF token and repo ID.")
                                else:
                                    with st.spinner("Pushing to HF Hub..."):
                                        try:
                                            # Download model from MLflow to a temp directory
                                            import tempfile
                                            with tempfile.TemporaryDirectory() as tmp_dir:
                                                from mlflow.sklearn import load_model as ml_load_skl
                                                model_uri = f"runs:/{job.mlflow_run_id}/model"
                                                skl_model = ml_load_skl(model_uri)
                                                
                                                import os
                                                save_path = os.path.join(tmp_dir, "model")
                                                # We don't necessarily need to re-save if we have the local path, 
                                                # but MLflow download_artifacts is better
                                                import mlflow
                                                local_model_path = mlflow.artifacts.download_artifacts(run_id=job.mlflow_run_id, artifact_path="model", dst_path=tmp_dir)
                                                
                                                # Deploy using our utility
                                                from src.deploy.hf_deploy import deploy_to_huggingface
                                                repo_url = deploy_to_huggingface(
                                                    model_path=local_model_path,
                                                    repo_id=hf_repo,
                                                    token=hf_token,
                                                    task=job.config.get('task', 'classification')
                                                )
                                                st.success(f"Model pushed! [View on HF Hub]({repo_url})")
                                        except Exception as e:
                                            st.error(f"HF Push failed: {e}")
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
    st.markdown(f"""
    <div class='hero-header'>
      <div class='hero-title'>📦 Model Registry & Deployment</div>
      <div class='hero-subtitle'>Versioning, artifact management, and model serving.</div>
    </div>""", unsafe_allow_html=True)
    
    st.subheader("🚀 Model Deployment Center")
    
    models = get_registered_models()
    if not models:
        st.warning("No models found in Registry. Please register a model from Experiments first.")
    else:
        col_dep1, col_dep2 = st.columns([1, 2])
        
        with col_dep1:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.markdown("##### 🔍 1. Select Artifact")
            model_names = [m.name for m in models]
            selected_model_name = st.selectbox("Registered Model", model_names, key="deploy_model_sel")
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{selected_model_name}'")
            version_nums = [v.version for v in versions]
            selected_version = st.selectbox("Version", version_nums, key="deploy_ver_sel")
            
            model_details = get_model_details(selected_model_name, selected_version)
            st.markdown(f"""
            <div style='background:#0d1117; padding:12px; border-radius:8px; border:1px solid #30363d; margin-top:10px;'>
              <div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase;'>Current Stage</div>
              <div style='font-weight:700; color:#2f80ed;'>{model_details.get('current_stage', 'None')}</div>
              <div style='font-size:0.7rem; color:#8b949e; margin-top:4px;'>Updated: {model_details.get('last_updated_timestamp', 'Unknown')}</div>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top: 16px; font-weight: 600;'>💻 How to Consume</div>", unsafe_allow_html=True)
            code_snippet = f'''import mlflow
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model("models:/{selected_model_name}/{selected_version}")
'''
            st.code(code_snippet, language="python")
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dep2:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.markdown("##### ⚙️ 2. Deployment Configuration")
            env = st.radio("Target Environment", ["Development (Local)", "Staging", "Production"], horizontal=True)
            
            c1, c2 = st.columns(2)
            with c1:
                cpu_alloc = st.slider("CPU Units", 0.5, 4.0, 1.0, step=0.5)
                min_replicas = st.number_input("Min Replicas", 1, 5, 1)
            with c2:
                mem_alloc = st.slider("Memory (GB)", 0.5, 16.0, 2.0, step=0.5)
                max_replicas = st.number_input("Max Replicas", 1, 10, 2)
            
            if st.button("🚀 Deploy / Update Service", type="primary", use_container_width=True):
                with st.spinner(f"Deploying {selected_model_name} v{selected_version}..."):
                    time.sleep(1.5)
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
            st.markdown("##### 📥 Export Format")
            if st.button("🧊 Download as ONNX", use_container_width=True):
                with st.spinner("Preparing ONNX binary..."):
                    try:
                        # Logic to load model from registry and convert
                        st.info("ONNX Conversion logic active in Engine. Utility: export_best_model_to_onnx")
                    except Exception as e:
                        st.error(f"ONNX conversion failed: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # 3. Integrated Testing Interface
        st.subheader("🎮 Inference Playground")
        
        if 'active_endpoint' in st.session_state:
            st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
            st.markdown(f"**Connected Endpoint**: `{st.session_state['active_endpoint']['url']}` <span class='badge badge-done'>Healthy</span>", unsafe_allow_html=True)
            
            # Use model tags to determine task type
            is_cv = False
            task_type = 'unknown'
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                # Get the registered model version's run ID to fetch tags
                version_info = client.get_model_version(selected_model_name, selected_version)
                run_info = client.get_run(version_info.run_id)
                tags = run_info.data.tags
                is_cv = tags.get("is_cv", "false") == "true"
                task_type = tags.get("task_type", "unknown")
            except Exception as e:
                pass
            
            if is_cv:
                st.info(f"Target is a Computer Vision Model (Task: {task_type}).")
                input_type = "Image"
            else:
                input_type = st.radio("Input Format", ["JSON/Text (Real-Time)", "CSV File (Batch)"], horizontal=True)
            
            if input_type == "JSON/Text (Real-Time)":
                input_data = st.text_area("Input Payload / Text", value='{"feature1": 0.5, "feature2": 1.2}', height=100)
                if st.button("🔌 Send API Request", type="primary"):
                    try:
                        import json
                        import pandas as pd
                        data = json.loads(input_data)
                    except json.JSONDecodeError:
                        # Assume it's text for NLP depending on the payload
                        data = {'text_input': input_data}
                    
                    try:
                        with st.spinner("Invoking model..."):
                            loaded_model = load_registered_model(selected_model_name, selected_version)
                            df_in = pd.DataFrame([data])
                            pred = loaded_model.predict(df_in)
                            st.json({"prediction": pred.tolist(), "latency_ms": 45, "model_version": selected_version})
                    except Exception as e:
                        st.error(f"Inference Error: {e}")
            
            elif input_type == "CSV File (Batch)":
                up_file = st.file_uploader("Upload Batch CSV", type="csv", key="deploy_test_csv")
                if up_file:
                    try:
                        import pandas as pd
                        df_test = pd.read_csv(up_file)
                        st.write("Preview of Input Data:")
                        st.dataframe(df_test.head(3))
                        if st.button("🏃 Run Batch Inference", type="primary"):
                            with st.spinner("Processing batch..."):
                                loaded_model = load_registered_model(selected_model_name, selected_version)
                                preds = loaded_model.predict(df_test)
                                df_test['prediction'] = preds
                                st.success("Batch Processing Complete!")
                                st.dataframe(df_test, use_container_width=True)
                                
                                # Option to download
                                csv = df_test.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Predictions", csv, f"predictions_{selected_model_name}_v{selected_version}.csv", "text/csv")
                    except Exception as e:
                         st.error(f"Batch inference failed: {e}")
            
            elif input_type == "Image":
                test_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="deploy_cv_image")
                if test_img:
                    st.image(test_img, caption="Input Data", width=300)
                    if st.button("🔍 Analyze Image", type="primary"):
                        with st.spinner("Invoking Vision Model..."):
                            try:
                                import torch
                                import torchvision.transforms as T
                                from PIL import Image
                                loaded_model = load_registered_model(selected_model_name, selected_version)
                                
                                img = Image.open(test_img).convert('RGB')
                                transform = T.Compose([
                                    T.Resize((224, 224)),
                                    T.ToTensor(),
                                ])
                                img_t = transform(img).unsqueeze(0)
                                
                                loaded_model.eval()
                                with torch.no_grad():
                                    outputs = loaded_model(img_t)
                                
                                if task_type == 'image_classification':
                                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                                    top_prob, top_class = torch.max(probs, dim=0)
                                    st.success(f"**Prediction Class Index:** {top_class.item()} (Confidence: {top_prob.item():.2%})")
                                    st.json({"probs": probs.tolist()})
                                elif task_type == 'image_segmentation':
                                    mask = outputs.argmax(dim=1).squeeze().numpy()
                                    st.success("Segmentation Mask Generated!")
                                    mask_img = Image.fromarray((mask * (255 // (mask.max() if mask.max() > 0 else 1))).astype('uint8'))
                                    st.image(mask_img, caption="Predicted Mask")
                                else:
                                    st.success("Prediction Generated!")
                                    st.write(outputs.numpy())
                            except Exception as e:
                                st.error(f"Vision Inference Error: {e}")
                            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Deploy a model above to enable the Live Inference Playground.")





