from automl_engine import AutoMLDataProcessor, AutoMLTrainer, save_pipeline, get_technical_explanation
from stability_engine import StabilityAnalyzer
from cv_engine import CVAutoMLTrainer, get_cv_explanation
import streamlit as st
import pandas as pd
import numpy as np
from mlops_utils import (
    MLFlowTracker, DriftDetector, ModelExplainer, get_model_registry, 
    DataLake, register_model_from_run, get_registered_models, get_all_runs,
    get_model_details, load_registered_model
)
import joblib # type: ignore
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import plotly.express as px
from PIL import Image

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

# 📊 Sidebar Metrics & Summary
with st.sidebar:
    st.title("🛡️ Platform Control")
    st.divider()
    
    # Quick Stats
    all_runs_df = get_all_runs()
    reg_models = get_registered_models()
    
    st.markdown("### 📈 System Overview")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Experiments", len(all_runs_df) if not all_runs_df.empty else 0)
    with col_s2:
        st.metric("Reg. Models", len(reg_models))
        
    st.divider()
    st.markdown("### 📁 Active Dataset")
    if 'df' in st.session_state:
        st.success(f"Rows: {st.session_state['df'].shape[0]}\nCols: {st.session_state['df'].shape[1]}")
    else:
        st.warning("No data loaded")

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
    "📊 Data", 
    "🤖 AutoML", 
    "🧪 Experiments", 
    "🖼️ Computer Vision", 
    "📈 Drift/Monitoring", 
    "🗂️ Model Registry",
    "🧪 Teste de Modelos",
    "⚖️ Estabilidade"
])

# --- TAB 0: DATA ---
with tabs[0]:
    st.header("📦 Data Lake & Management")
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

# --- TAB 1: AUTOML & MODEL HUB ---
with tabs[1]:
    st.header("🤖 AutoML & Model Hub")
    
    # --- SUB-TAB: NOVO TREINO (UNIFICADO) ---
    st.subheader("📋 Configuração do Treino")
    
    # 1. Definição da Tarefa
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        task = st.radio("Tipo de Tarefa", ["classification", "regression", "clustering", "time_series", "anomaly_detection"], key="task_selector_train")
    
    with col_t2:
        training_strategy = st.radio("Configuração de Hiperparâmetros", ["Automático", "Manual"], 
                                     help="Automático: O sistema busca os melhores parâmetros. Manual: Você define tudo.")

    st.divider()

    # 2. Configuração de Modelos e Parâmetros
    st.subheader("🎯 Seleção do Modelo")
    
    # Seletor de Fonte do Modelo (Migrado de Fine-Tune)
    model_source = st.radio("Fonte do Modelo", 
                           ["AutoML Standard (Scikit-Learn/XGBoost/Transformers)", 
                            "Model Registry (Registrados)", 
                            "Upload Local (.pkl)"],
                           horizontal=True)

    trainer_temp = AutoMLTrainer(task_type=task)
    available_models = trainer_temp.get_available_models()
    
    selected_models = None
    manual_params = None
    
    # Lógica de Seleção Baseada na Fonte
    if model_source == "AutoML Standard (Scikit-Learn/XGBoost/Transformers)":
        mode_selection = st.radio("Seleção de Modelos", ["Automático (Preset)", "Manual (Selecionar)"], horizontal=True)
        if mode_selection == "Manual (Selecionar)":
            selected_models = st.multiselect("Escolha os Modelos", available_models, default=available_models[:2] if available_models else None)
    
    elif model_source == "Model Registry (Registrados)":
        reg_models = get_registered_models()
        if reg_models:
            base_model_name = st.selectbox("Selecione o Modelo Registrado", [m.name for m in reg_models], key="reg_sel_train")
            selected_models = [base_model_name]
            st.info(f"O modelo '{base_model_name}' será usado como base para retreino/fine-tune.")
        else:
            st.warning("Nenhum modelo registrado encontrado.")

    elif model_source == "Upload Local (.pkl)":
        uploaded_pkl = st.file_uploader("Upload do arquivo .pkl base", type="pkl", key="pkl_upload_train")
        if uploaded_pkl:
            selected_models = ["Uploaded_Model"] # Placeholder, precisaria de lógica customizada no backend
            st.info("Modelo carregado para retreino.")

    st.subheader("🎯 Configuração da Otimização")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        # Seletor unificado de preset (incluindo 'custom')
        if model_source == "AutoML Standard (Scikit-Learn/XGBoost/Transformers)":
            training_preset = st.select_slider(
                "Modo de Treinamento (Preset)",
                options=["fast", "medium", "best_quality", "custom"],
                value="medium",
                help="fast: Rápido. medium: Equilibrado. best_quality: Exaustivo. custom: Defina suas regras."
            )
        else:
            # Para outros modos, permitimos customizar mas iniciamos com medium
            st.info(f"Modo base adaptado para {model_source}")
            # Aqui podemos permitir customizar n_trials também
            use_custom_tuning = st.checkbox("Customizar Otimização (Trials/Timeout)", value=False)
            training_preset = "custom" if use_custom_tuning else "medium"

        # Inputs condicionais para modo custom
        if training_preset == "custom":
            st.markdown("##### 🛠️ Configuração Customizada")
            n_trials = st.number_input("Número de Tentativas (por modelo)", 1, 1000, 20, key="cust_trials")
            timeout_per_model = st.number_input("Timeout por modelo (segundos)", 10, 7200, 600, key="cust_timeout")
            early_stopping = st.number_input("Early Stopping (Rounds)", 0, 50, 7, key="cust_es")
            
            st.markdown("##### ⚡ Parâmetros Avançados")
            custom_max_iter = st.number_input("Máximo de Iterações (max_iter)", 100, 100000, 1000, help="Limite de iterações para solvers (LogisticRegression, SVM, MLP). Valores muito altos podem causar lentidão.")
            
            # NLP Configuration (if applicable)
            st.markdown("##### 📝 NLP Avançado")
            nlp_max_features = st.number_input("Max Features (Vetorização)", min_value=100, max_value=None, value=20000, step=1000, help="Número máximo de features para TF-IDF/CountVectorizer. Deixe alto para capturar mais vocabulário (ex: 20000+). Otimizado automaticamente se for muito alto.")
            nlp_ngram_range_max = st.slider("N-Gram Range Max", 1, 3, 2, help="Tamanho máximo dos n-grams (1=unigramas, 2=bigramas, 3=trigramas).")
            
            manual_params = {
                'max_iter': custom_max_iter,
                'nlp_max_features': nlp_max_features,
                'nlp_ngram_range': (1, nlp_ngram_range_max)
            }
        else:
            n_trials = None
            timeout_per_model = None
            early_stopping = 10
            manual_params = {}

        
        with col_opt2:
            st.markdown("##### 🛡️ Estratégia de Validação")
            validation_options = ["K-Fold Cross Validation", "Stratified K-Fold", "Holdout (Treino/Teste)", "Auto-Split (Otimizado)", "Time Series Split"]
            
            # Filtrar opções baseadas na tarefa
            if task == "time_series":
                val_strategy_ui = "Time Series Split"
                st.info("Séries temporais usam divisão temporal obrigatoriamente.")
                validation_strategy = 'time_series_cv'
            elif task == "classification":
                val_strategy_ui = st.selectbox("Método de Validação", validation_options, index=0)
            else: # regression, clustering, anomaly
                # Stratified só faz sentido para classificação
                opts = [o for o in validation_options if o != "Stratified K-Fold"]
                val_strategy_ui = st.selectbox("Método de Validação", opts, index=0)
            
            validation_params = {}
            if val_strategy_ui in ["K-Fold Cross Validation", "Stratified K-Fold"]:
                n_folds = st.number_input("Número de Folds", 2, 20, 5, key="val_folds")
                validation_params['folds'] = n_folds
                validation_strategy = 'cv' if val_strategy_ui == "K-Fold Cross Validation" else 'stratified_cv'
            elif val_strategy_ui == "Holdout (Treino/Teste)":
                test_size = st.slider("Tamanho do Teste (%)", 10, 50, 20, key="val_holdout") / 100.0
                validation_params['test_size'] = test_size
                validation_strategy = 'holdout'
            elif val_strategy_ui == "Auto-Split (Otimizado)":
                st.info("O sistema decidirá o melhor split durante a otimização.")
                validation_strategy = 'auto_split'
            elif val_strategy_ui == "Time Series Split":
                n_splits = st.number_input("Número de Splits Temporais", 2, 20, 5, key="val_ts_splits")
                validation_params['folds'] = n_splits
                validation_strategy = 'time_series_cv'
            
            # Seleção de colunas NLP
            st.markdown("##### 🔤 Configuração de NLP")
            
            # Configurações Avançadas de NLP
            # Usamos um container para renderizar as opções de NLP mais tarde,
            # assim que tivermos acesso ao sample_df (preview dos dados).
            nlp_container = st.container()
            nlp_config_automl = {} 

            if task == "time_series":
                st.info("💡 Split temporal obrigatório para séries temporais.")

        st.divider()
        st.subheader("🌱 Configuração de Reprodutibilidade (Seed)")
        seed_mode = st.radio("Modo de Seed", 
                             ["Automático (Diferente por modelo)", 
                              "Automático (Mesma para todos)", 
                              "Manual (Mesma para todos)", 
                              "Manual (Diferente por modelo)"], 
                             horizontal=True)
        
        random_seed_config = 42 # Default
        
        effective_models = selected_models if selected_models else available_models
        
        if seed_mode == "Automático (Diferente por modelo)":
            random_seed_config = {m: np.random.randint(0, 999999) for m in effective_models}
            st.info("🎲 Seeds aleatórias serão geradas para cada modelo.")
        elif seed_mode == "Automático (Mesma para todos)":
            random_seed_config = np.random.randint(0, 999999)
            st.info(f"🎲 Uma única seed aleatória será usada para todos: {random_seed_config}")
        elif seed_mode == "Manual (Mesma para todos)":
            random_seed_config = st.number_input("🌱 Digite a Seed Global", 0, 999999, 42)
        elif seed_mode == "Manual (Diferente por modelo)":
            st.markdown("##### Digite a Seed para cada modelo:")
            random_seed_config = {}
            cols_seed = st.columns(min(len(effective_models), 3))
            for i, m in enumerate(effective_models):
                with cols_seed[i % 3]:
                    random_seed_config[m] = st.number_input(f"Seed: {m}", 0, 999999, 42, key=f"seed_{m}")

        # Hiperparâmetros Manuais integrados nas opções de tuning
        if training_strategy == "Manual":
            st.divider()
            st.subheader("⚙️ Configuração de Hiperparâmetros Manuais")
            st.info("Nota: No modo Manual, você define os parâmetros que serão usados como ponto de partida (enqueue) para os modelos selecionados.")
            
            # Se múltiplos modelos estiverem selecionados, o usuário pode configurar um por um ou um modelo de referência
            ref_model = st.selectbox("Modelo para Configurar", selected_models or available_models)
            
            # Merge existing manual_params with new manual config
            current_manual_params = manual_params.copy()
            current_manual_params['model_name'] = ref_model
            
            schema = trainer_temp.get_model_params_schema(ref_model)

            if schema:
                st.markdown(f"**Parâmetros para {ref_model}**")
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
        st.subheader("📂 Seleção de Dados")
        available_datasets = datalake.list_datasets()
        selected_ds_list = st.multiselect("Escolha os Datasets", available_datasets, key="ds_train_multi")
        
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
                        if task not in ["clustering", "anomaly_detection"]:
                            target_pre = st.selectbox("🎯 Target (Variável Alvo)", sample_df.columns, key="target_selector_pre")
                    
                    with col_sel2:
                        if task == "time_series":
                            date_col_pre = st.selectbox("📅 Coluna de Data (OBRIGATÓRIO)", sample_df.columns, key="ts_date_selector")
            except Exception as e: st.error(f"Erro ao carregar amostra: {e}")

        selected_configs = []
        if selected_ds_list:
            cols_ds = st.columns(len(selected_ds_list))
            for i, ds_name in enumerate(selected_ds_list):
                with cols_ds[i]:
                    st.markdown(f"**{ds_name}**")
                    versions = datalake.list_versions(ds_name)
                    ver = st.selectbox(f"Versão", versions, key=f"ver_{ds_name}")
                    
                    # Configuração de Papel do Dataset (Granularidade Solicitada)
                    if validation_strategy == 'holdout':
                        st.caption("Defina como usar este dataset:")
                        role = st.radio("Papel", ["Treino + Teste (Split)", "Apenas Treino (100%)", "Apenas Teste (100%)"], key=f"role_{ds_name}")
                        
                        split = 100
                        if role == "Treino + Teste (Split)":
                            split = st.slider(f"% Treino", 10, 95, 80, key=f"split_{ds_name}")
                        elif role == "Apenas Teste (100%)":
                            split = 0
                    else:
                        # Para estratégias como K-Fold ou Auto-Split, usamos o dataset integralmente no processo (split=100)
                        # O sistema de validação cuidará da divisão interna.
                        split = 100
                        st.info(f"Dataset usado integralmente para {validation_strategy}")
                    
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
                selected_nlp_cols = st.multiselect("Colunas de Texto (NLP)", potential_nlp_cols, help="Selecione as colunas que contêm texto para processamento NLP otimizado.")
                
                if selected_nlp_cols:
                    col_nlp1, col_nlp2 = st.columns(2)
                    with col_nlp1:
                        vectorizer_automl = st.selectbox("Vetorização", ["tfidf", "count"], key="automl_vect")
                        ngram_min_automl, ngram_max_automl = st.slider("N-Grams Range", 1, 3, (1, 2), key="automl_ngram")
                    with col_nlp2:
                        remove_stopwords_automl = st.checkbox("Remover Stopwords (English)", value=True, key="automl_stop")
                        lematization_automl = st.checkbox("Lematização (WordNet - requer NLTK)", value=False, key="automl_lemma")
                        max_features_automl = st.number_input("Max Features", min_value=100, max_value=None, value=20000, step=1000, key="automl_max_feat", help="Deixe alto (ex: 20000) para capturar mais vocabulário. Otimizado automaticamente.")

                    nlp_config_automl = {
                        "vectorizer": vectorizer_automl,
                        "ngram_range": (ngram_min_automl, ngram_max_automl),
                        "stop_words": remove_stopwords_automl,
                        "max_features": max_features_automl,
                        "lemmatization": lematization_automl
                    }
            else:
                if selected_ds_list:
                    st.info("Nenhuma coluna de texto identificada na amostra.")
                else:
                    st.info("Selecione um dataset abaixo para configurar NLP.")

        if selected_configs:
            if st.button("📥 Carregar e Preparar Dados", key="btn_load_train"):
                # Usar configurações individuais de split (global_split=None)
                train_df, test_df = prepare_multi_dataset(selected_configs, global_split=None, task_type=task, date_col=date_col_pre, target_col=target_pre)
                
                st.session_state['train_df'] = train_df
                st.session_state['test_df'] = test_df
                st.session_state['current_task'] = task
                st.session_state['date_col_active'] = date_col_pre
                st.session_state['target_active'] = target_pre # Salvar target selecionado
                st.session_state['n_trials_active'] = n_trials
                st.session_state['early_stopping_active'] = early_stopping
                st.success("Dados carregados!")

        if 'train_df' in st.session_state and st.session_state.get('current_task') == task:
            train_df = st.session_state['train_df']
            test_df = st.session_state['test_df']
            
            st.divider()
            st.subheader("⚙️ Configuração Final")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if task not in ["clustering", "anomaly_detection"]:
                    # Se já foi selecionado no pré-carregamento, apenas exibir e travar
                    if st.session_state.get('target_active') and st.session_state['target_active'] in train_df.columns:
                        target = st.session_state['target_active']
                        st.info(f"🎯 Target Definido: **{target}** (Para alterar, recarregue os dados)")
                    else:
                        target = st.selectbox("🎯 Selecione o Target", train_df.columns)
                else:
                    target = None
            
            with col_f2:
                if task == "time_series":
                    freq = st.selectbox("⏱️ Intervalo", ["Minutos", "Horas", "Dias", "Semanas", "Meses", "Anos"])
                    forecast_horizon = st.number_input("🔮 Horizonte", 1, 100, 7)
                else: forecast_horizon, freq = 1, "D"

            if st.button("🚀 Iniciar Treinamento", key="btn_start_train"):
                st.session_state['trials_data'] = []
                start_time_train = time.time()
                
                # Nome do experimento baseado no dataset e timestamp
                exp_tag = selected_configs[0]['name'] if selected_configs else "AutoML"
                experiment_name = f"{exp_tag}_{task}_{time.strftime('%Y%m%d_%H%M%S')}"

                # Containers for feedback
                status_c = st.empty()
                progress_bar = st.progress(0)
                chart_c = st.empty()
                
                # Calcular total real de trials para a barra de progresso
                # Instancia o trainer com o preset para pegar as configs
                trainer_for_info = AutoMLTrainer(task_type=task, preset=training_preset)
                preset_config = trainer_for_info.preset_configs.get(training_preset)
                
                effective_models_list = selected_models if selected_models else preset_config['models']
                n_trials_val = n_trials if n_trials is not None else preset_config['n_trials']
                total_expected_trials = n_trials_val * len(effective_models_list)
                
                def callback(trial, score, full_name, dur, metrics=None):
                    # Extrair nome do algoritmo e o número do trial do modelo
                    algo_name = full_name.split(" - ")[0]
                    trial_label = full_name.split(" - ")[1] # "Trial X"
                    trial_num = int(trial_label.replace("Trial ", ""))

                    trial_info = {
                        "Tentativa Geral": trial.number + 1,
                        "Trial Modelo": trial_num,
                        "Modelo": algo_name,
                        "Identificador": full_name,
                        "Score": score,
                        "Duração (s)": dur
                    }
                    
                    # Adicionar outras métricas ao dicionário do trial
                    if metrics:
                        for m_name, m_val in metrics.items():
                            if m_name != 'confusion_matrix' and isinstance(m_val, (int, float, np.number)):
                                trial_info[m_name.upper()] = m_val

                    st.session_state['trials_data'].append(trial_info)
                    
                    df_trials = pd.DataFrame(st.session_state['trials_data'])
                    
                    with status_c:
                        metric_text = f"Score: {score:.4f}"
                        if metrics:
                            # Mostrar a métrica principal de forma destacada
                            main_metric = next(iter(metrics))
                            metric_text = f"{main_metric.upper()}: {metrics[main_metric]:.4f}"
                            
                        st.info(f"✨ {full_name} concluído | {metric_text} | Total: {trial.number + 1}/{total_expected_trials}")
                    
                    progress_bar.progress(min((trial.number + 1) / total_expected_trials, 1.0))
                    
                    with chart_c:
                        # Gráfico mostrando o progresso de cada modelo individualmente
                        # Determinar o nome da métrica principal para o eixo Y
                        main_metric_name = "Métrica"
                        if metrics:
                            main_metric_name = next(iter(metrics)).upper()
                        
                        fig = px.line(df_trials, x="Trial Modelo", y="Score", color="Modelo", 
                                    markers=True, hover_name="Identificador",
                                    title="Progresso da Otimização por Algoritmo")
                        fig.update_layout(xaxis_title="Nº da Tentativa do Modelo", yaxis_title=f"Score ({main_metric_name})")
                        st.plotly_chart(fig, key=f"chart_{trial.number}", use_container_width=True)

                with st.spinner("Processando..."):
                    processor = AutoMLDataProcessor(target_column=target, task_type=task, date_col=date_col_pre, forecast_horizon=forecast_horizon, nlp_config=nlp_config_automl)
                    X_train_proc, y_train_proc = processor.fit_transform(train_df, nlp_cols=selected_nlp_cols)
                    X_test_proc, y_test_proc = processor.transform(test_df) if test_df is not None else (None, None)
                    
                    # Preparar modelos customizados (Upload/Registry)
                    custom_models = {}
                    if model_source == "Upload Local (.pkl)" and 'uploaded_pkl' in locals() and uploaded_pkl:
                         try:
                             loaded_model = joblib.load(uploaded_pkl)
                             custom_models["Uploaded_Model"] = loaded_model
                         except Exception as e:
                             st.error(f"Erro ao carregar .pkl: {e}")
                             st.stop()
                    elif model_source == "Model Registry (Registrados)" and selected_models:
                         model_name = selected_models[0]
                         try:
                             loaded_model = load_registered_model(model_name)
                             custom_models[model_name] = loaded_model
                         except Exception as e:
                             st.error(f"Erro ao carregar do registry: {e}")
                             st.stop()

                    trainer = AutoMLTrainer(task_type=task, preset=training_preset)
                    
                    best_model = trainer.train(
                        X_train_proc, 
                        y_train_proc, 
                        n_trials=n_trials,
                        timeout=timeout_per_model,
                        callback=callback, 
                        selected_models=selected_models, 
                        early_stopping_rounds=early_stopping,
                        manual_params=manual_params,
                        experiment_name=experiment_name,
                        random_state=random_seed_config,
                        validation_strategy=validation_strategy,
                        validation_params=validation_params,
                        custom_models=custom_models
                    )
                    best_params = trainer.best_params
                    
                    st.session_state['best_model'] = best_model
                    st.session_state['best_params'] = best_params
                    st.session_state['processor'] = processor
                    
                    # Evaluation
                    metrics, y_pred = trainer.evaluate(X_test_proc, y_test_proc) if X_test_proc is not None else (None, None)
                    
                    st.success("🎉 Processo de AutoML Finalizado com Sucesso!")
                    
                    # Mostrar o melhor modelo de forma destacada
                    best_model_name = trainer.best_params.get('model_name', 'Desconhecido')
                    st.balloons()
                    st.markdown(f"""
                        <div style="background-color:#d4edda; padding:20px; border-radius:10px; border-left:8px solid #28a745; margin-bottom:20px;">
                            <h2 style="color:#155724; margin:0;">🏆 Melhor Modelo Encontrado: {best_model_name}</h2>
                            <p style="color:#155724; font-size:1.1em; margin-top:10px;">O sistema otimizou e selecionou o algoritmo acima como o de melhor performance para sua tarefa.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # --- Resumo por Modelo ---
                    if hasattr(trainer, 'model_summaries') and trainer.model_summaries:
                        st.markdown("### 🏆 Melhores Resultados por Algoritmo")
                        summary_data = []
                        for m_name, info in trainer.model_summaries.items():
                            row = {
                                "Algoritmo": m_name,
                                "Melhor Score": f"{info['score']:.4f}",
                                "Trial": info['trial_name'],
                                "Duração (s)": f"{info['duration']:.2f}"
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
                        with st.expander("📋 Ver Histórico Completo de Todas as Tentativas"):
                            df_all = pd.DataFrame(st.session_state['trials_data'])
                            st.dataframe(df_all.sort_values(by="Score", ascending=False), use_container_width=True)

                    if metrics: 
                        st.markdown("### 📊 Resultados Finais (Melhor Modelo Global)")
                        cols_m = st.columns(len(metrics))
                        for i, (m_name, m_val) in enumerate(metrics.items()):
                            if m_name != 'confusion_matrix':
                                with cols_m[i % len(cols_m)]:
                                    st.metric(m_name.upper(), f"{m_val:.4f}" if isinstance(m_val, (float, np.float64, np.float32)) else m_val)
                    
                    # --- Visualizações de Resultados ---
                    if X_test_proc is not None:
                        st.divider()
                        st.subheader("📈 Visualização de Performance")
                        
                        if task == "classification":
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                if 'confusion_matrix' in metrics:
                                    cm = np.array(metrics['confusion_matrix'])
                                    fig_cm = px.imshow(cm, text_auto=True, title="Matriz de Confusão",
                                                     labels=dict(x="Predito", y="Real", color="Quantidade"))
                                    st.plotly_chart(fig_cm)
                            with col_v2:
                                # Feature Importance
                                if hasattr(trainer, 'feature_importance') and trainer.feature_importance:
                                    st.markdown("#### 📈 Importância das Features")
                                    fi_data = pd.DataFrame({
                                        'Feature': processor.get_feature_names(),
                                        'Importância': trainer.feature_importance
                                    }).sort_values(by='Importância', ascending=False)
                                    
                                    fig_fi = px.bar(fi_data.head(15), x='Importância', y='Feature', orientation='h',
                                                  title="Top 15 Features mais Importantes")
                                    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig_fi, use_container_width=True)
                                else:
                                    # Fallback for complex models
                                    st.info("Calculando importância das features via SHAP...")
                                    try:
                                        explainer = ModelExplainer(best_model, X_train_proc[:100])
                                        st.pyplot(explainer.plot_importance(X_test_proc[:100]))
                                    except:
                                        st.warning("Não foi possível gerar SHAP plot para este modelo.")

                        elif task in ["regression", "time_series"]:
                            df_res = pd.DataFrame({"Real": y_test_proc, "Predito": y_pred})
                            if task == "time_series":
                                fig_res = px.line(df_res.reset_index(), y=["Real", "Predito"], title="Série Temporal: Real vs Predito")
                            else:
                                fig_res = px.scatter(df_res, x="Real", y="Predito", trendline="ols", title="Regressão: Real vs Predito")
                            st.plotly_chart(fig_res)

                        elif task == "clustering":
                            # PCA for visualization
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_test_proc)
                            df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                            df_pca['Cluster'] = y_pred.astype(str)
                            fig_cluster = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title="Visualização de Clusters (PCA)")
                            st.plotly_chart(fig_cluster)

                        elif task == "anomaly_detection":
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_test_proc)
                            df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                            # y_pred: -1 for anomaly, 1 for normal
                            df_pca['Status'] = np.where(y_pred == -1, 'Anomalia', 'Normal')
                            fig_anom = px.scatter(df_pca, x='PCA1', y='PCA2', color='Status', 
                                                color_discrete_map={'Anomalia': 'red', 'Normal': 'blue'},
                                                title="Detecção de Anomalias (PCA)")
                            st.plotly_chart(fig_anom)



# --- TAB 2: EXPERIMENTS ---
with tabs[2]:
    st.header("🧪 Experiments Explorer")
    st.markdown("Aqui você encontra o histórico de **todos os treinos**. Escolha os melhores para registrar no catálogo oficial.")
    
    runs = get_all_runs()
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
            run_id_sel = st.selectbox("🔍 Select Run to Explore", filtered_runs['run_id'].tolist())
            if run_id_sel:
                run_data = filtered_runs[filtered_runs['run_id'] == run_id_sel].iloc[0]
                st.markdown("#### 📊 Metrics")
                metrics = {k.replace('metrics.', ''): v for k, v in run_data.items() if k.startswith('metrics.') and pd.notna(v)}
                st.json(metrics)
        
        with col_det2:
            if run_id_sel:
                st.markdown("#### 🚀 Register as Official Model")
                model_reg_name = st.text_input("Registry Name", value=f"model_{run_id_sel[:6]}")
                if st.button("Confirm Registration"):
                    if register_model_from_run(run_id_sel, model_reg_name):
                        st.success(f"Model {model_reg_name} is now in the Registry!")
                        st.rerun()
    else:
        st.info("Nenhum experimento encontrado. Inicie um treino na aba AutoML & Model Hub.")

# --- TAB 3: COMPUTER VISION ---
with tabs[3]:
    st.header("🖼️ Computer Vision AutoML")
    cv_task = st.selectbox("CV Task", ["image_classification", "image_segmentation", "object_detection"])
    
    col_cv1, col_cv2 = st.columns(2)
    with col_cv1:
        data_dir = st.text_input("Dataset Directory", "data/images/classification")
        if cv_task == "image_segmentation":
            mask_dir = st.text_input("Masks Directory (for Segmentation)", "data/images/masks")
        elif cv_task == "object_detection":
            mask_dir = st.text_input("Annotations Directory (for Detection)", "data/images/annotations")
        else:
            mask_dir = None
            
    with col_cv2:
        epochs = st.number_input("Epochs", 1, 100, 5)
        lr_cv = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")

    if st.button("🚀 Start CV Training"):
        trainer = CVAutoMLTrainer(task_type=cv_task)
        
        status_cv = st.empty()
        progress_cv = st.progress(0)
        
        def cv_callback(epoch, acc, loss, duration):
            status_cv.write(f"Epoch {epoch}: Acc={acc:.4f}, Loss={loss:.4f}, Time={duration:.2f}s")
            progress_cv.progress((epoch + 1) / epochs)

        with st.spinner("Training vision model..."):
            best_model_cv = trainer.train(data_dir, n_epochs=epochs, lr=lr_cv, callback=cv_callback, mask_dir=mask_dir)
            st.success("Vision Training Complete!")
            st.session_state['best_cv_model'] = best_model_cv
            st.session_state['cv_trainer'] = trainer

    if st.session_state.get('best_cv_model'):
        st.divider()
        st.subheader("Inference Test")
        test_img = st.file_uploader("Upload image for prediction", type=['jpg', 'png'])
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
                    st.metric("Predicted Class ID", prediction)
            
            os.remove(img_path)

# --- TAB 4: DRIFT / MONITORING ---
with tabs[4]:
    st.header("📈 Data Drift & Monitoring")
    if 'df' in st.session_state:
        ref_df = st.session_state['df']
        curr_file = st.file_uploader("Upload Current Data for Drift Analysis", type="csv")
        if curr_file:
            curr_df = pd.read_csv(curr_file)
            detector = DriftDetector()
            drifts = detector.detect_drift(ref_df, curr_df)
            st.write("Drift Results:")
            st.json(drifts)
            drift_detected = any(d['drift_detected'] for d in drifts.values())
            if drift_detected:
                st.error("🚨 Drift Detected! Retrain recommended.")
            else:
                st.success("✅ No drift detected.")
    else:
        st.warning("Please upload reference data in the Data tab.")

# --- TAB 5: MODEL REGISTRY ---
with tabs[5]:
    st.header("🗂️ Official Model Registry")
    st.markdown("Apenas modelos validados e registrados manualmente via aba Experiments.")
    
    models = get_registered_models()
    if models:
        for m in models:
            with st.expander(f"📦 {m.name}"):
                st.write(f"**Last Modified:** {m.last_updated_timestamp}")
                st.write(f"**Description:** {m.description or 'No description provided'}")
                if st.button(f"Deploy {m.name}", key=f"deploy_{m.name}"):
                    st.success(f"Deployment pipeline started for {m.name}!")
    else:
        st.warning("Nenhum modelo registrado no catálogo oficial ainda.")

# --- TAB 6: TESTE DE MODELOS ---
with tabs[6]:
    st.header("🧪 Teste de Modelos")
    st.markdown("Teste modelos registrados ou faça upload de um arquivo de modelo local (.pkl, .joblib).")
    
    test_mode = st.radio("Origem do Modelo", ["Model Registry", "Upload Local"], horizontal=True)
    
    # Adicionar botão para limpar o modelo atual do teste
    if 'test_model' in st.session_state:
        if st.button("🗑️ Limpar Modelo Carregado"):
            del st.session_state['test_model']
            if 'test_metadata' in st.session_state: del st.session_state['test_metadata']
            st.rerun()

    if test_mode == "Model Registry":
        reg_models = get_registered_models()
        if reg_models:
            model_names = [m.name for m in reg_models]
            sel_model_name = st.selectbox("Selecione o Modelo Registrado", model_names)
            
            # Pegar versões do modelo
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = [v.version for v in client.search_model_versions(f"name='{sel_model_name}'")]
            sel_version = st.selectbox("Versão", versions)
            
            if st.button("Carregar Modelo do Registry"):
                with st.spinner("Carregando modelo e metadados..."):
                    try:
                        loaded_model = load_registered_model(sel_model_name, sel_version)
                        if loaded_model is not None:
                            st.session_state['test_model'] = loaded_model
                            st.session_state['test_metadata'] = get_model_details(sel_model_name, sel_version)
                            st.success(f"Modelo {sel_model_name} (v{sel_version}) carregado!")
                            st.rerun()
                        else:
                            st.error("Falha ao carregar o objeto do modelo do Registry.")
                    except Exception as e:
                        st.error(f"Erro ao carregar modelo: {e}")
        else:
            st.warning("Nenhum modelo registrado encontrado.")
            
    else:
        uploaded_model = st.file_uploader("Upload do arquivo do modelo (.pkl, .joblib)", type=["pkl", "joblib"])
        if uploaded_model:
            if st.button("Carregar Modelo Uploaded"):
                try:
                    if uploaded_model.name.endswith(".pkl"):
                        loaded_model = pickle.load(uploaded_model)
                    else:
                        loaded_model = joblib.load(uploaded_model)
                    
                    if loaded_model is not None:
                        st.session_state['test_model'] = loaded_model
                        st.session_state['test_metadata'] = {"name": uploaded_model.name, "version": "Local", "params": "N/A", "source": "Upload"}
                        st.success("Modelo local carregado com sucesso!")
                        st.rerun()
                    else:
                        st.error("O arquivo carregado resultou em um objeto nulo.")
                except Exception as e:
                    st.error(f"Erro ao carregar modelo: {e}")

    # Exibição de Metadados e Teste de Previsão
    if 'test_model' in st.session_state:
        model = st.session_state['test_model']
        meta = st.session_state['test_metadata']
        
        st.divider()
        col_m1, col_m2 = st.columns([1, 2])
        
        with col_m1:
            st.subheader("📋 Informações do Modelo")
            st.write(f"**Nome:** {meta.get('name')}")
            st.write(f"**Versão:** {meta.get('version')}")
            st.write(f"**Fonte:** {meta.get('source', 'Registry')}")
            
            if 'params' in meta and meta['params'] != "N/A":
                with st.expander("⚙️ Parâmetros"):
                    st.json(meta['params'])
            
            if 'metrics' in meta:
                with st.expander("📊 Métricas de Treino"):
                    st.json(meta['metrics'])
                    
        with col_m2:
            st.subheader("🔮 Realizar Previsão")
            
            test_input_mode = st.radio("Entrada de Dados", ["Manual (JSON/Campos)", "Upload CSV"], horizontal=True)
            
            prediction_result = None
            
            if test_input_mode == "Upload CSV":
                test_csv = st.file_uploader("Upload CSV para Previsão", type="csv", key="test_csv_upload")
                if test_csv:
                    test_df = pd.read_csv(test_csv)
                    st.write("Preview dos Dados:", test_df.head(3))
                    
                    if st.button("🚀 Gerar Previsões"):
                        try:
                            # Tentar usar o processador se disponível no session_state (opcional)
                            if 'processor' in st.session_state:
                                proc = st.session_state['processor']
                                # Garantir que o target não esteja no CSV de teste para o transform
                                if proc.target_column in test_df.columns:
                                    X_test = test_df.drop(columns=[proc.target_column])
                                else:
                                    X_test = test_df
                                X_proc, _ = proc.transform(X_test)
                                preds = model.predict(X_proc)
                            else:
                                preds = model.predict(test_df)
                                
                            test_df['PREDICTION'] = preds
                            st.write("Resultados:")
                            st.dataframe(test_df.head(10))
                            
                            csv = test_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Baixar Resultados (CSV)", csv, "predictions.csv", "text/csv")
                        except Exception as e:
                            st.error(f"Erro na previsão: {e}. Verifique se os dados de entrada possuem as mesmas colunas do treino.")
            
            else:
                # Entrada manual - Tenta inferir colunas
                cols_to_input = []
                if 'processor' in st.session_state:
                    cols_to_input = [c for c in st.session_state['processor'].feature_columns]
                elif hasattr(model, "feature_names_in_"):
                    cols_to_input = list(model.feature_names_in_)
                elif hasattr(model, "feature_names"): # Para alguns modelos como CatBoost/XGBoost
                    cols_to_input = list(model.feature_names)
                
                if not cols_to_input and test_input_mode == "Manual (JSON/Campos)":
                    st.warning("⚠️ Não foi possível detectar as colunas automaticamente. Você pode colar um JSON com as características abaixo:")
                    json_input = st.text_area("JSON de entrada (ex: {'feat1': 10, 'feat2': 20})", value="{}")
                    
                    if st.button("🚀 Prever (via JSON)"):
                        try:
                            import json
                            data = json.loads(json_input)
                            input_df = pd.DataFrame([data])
                            
                            if 'processor' in st.session_state:
                                X_proc, _ = st.session_state['processor'].transform(input_df)
                                pred = model.predict(X_proc)
                            else:
                                pred = model.predict(input_df)
                            st.success(f"Resultado da Previsão: **{pred[0]}**")
                        except Exception as e:
                            st.error(f"Erro no JSON ou Previsão: {e}")
                
                elif cols_to_input:
                    st.info("Insira os valores para cada característica:")
                    input_data = {}
                    col_idx = 0
                    cols_layout = st.columns(3)
                    for col_name in cols_to_input:
                        with cols_layout[col_idx % 3]:
                            input_data[col_name] = st.text_input(col_name, value="0")
                        col_idx += 1
                    
                    if st.button("🚀 Prever (Manual)"):
                        try:
                            # Converter para DataFrame de uma linha
                            input_df = pd.DataFrame([input_data])
                            # Converter tipos se possível (tentar float)
                            for c in input_df.columns:
                                try:
                                    input_df[c] = pd.to_numeric(input_df[c])
                                except: pass
                            
                            if 'processor' in st.session_state:
                                X_proc, _ = st.session_state['processor'].transform(input_df)
                                pred = model.predict(X_proc)
                            else:
                                pred = model.predict(input_df)
                                
                            st.success(f"Resultado da Previsão: **{pred[0]}**")
                        except Exception as e:
                            st.error(f"Erro: {e}")
                else:
                    st.warning("Não foi possível identificar as colunas necessárias. Use o Upload CSV ou certifique-se de que o modelo foi treinado nesta sessão.")

# --- TAB 7: ESTABILIDADE ---
with tabs[7]:
    st.header("⚖️ Análise de Estabilidade e Robustez Avançada")
    st.markdown("Avaliação de consistência do modelo com configurações de MLOps, NLP e validação cruzada.")

    col_config, col_main = st.columns([1, 2])

    with col_config:
        st.subheader("⚙️ Configuração do Experimento")
        
        # --- 1. DADOS ---
        st.markdown("### 1. 📂 Dados e Tarefa")
        dataset_names = datalake.list_datasets()
        if not dataset_names:
            st.warning("Nenhum dataset no Data Lake.")
            selected_ds = None
            df_stab = None
        else:
            selected_ds = st.selectbox("Dataset", dataset_names, key="stab_ds")
            if selected_ds:
                versions = datalake.list_versions(selected_ds)
                selected_ver = st.selectbox("Versão", versions, key="stab_ver")
                
                try:
                    df_stab = datalake.load_version(selected_ds, selected_ver)
                    st.success(f"Carregado: {len(df_stab)} linhas")
                    
                    all_cols = df_stab.columns.tolist()
                    default_target = all_cols[-1]
                    if 'target' in all_cols: default_target = 'target'
                    if 'class' in all_cols: default_target = 'class'
                    
                    target_col = st.selectbox("Coluna Alvo (Target)", all_cols, index=all_cols.index(default_target) if default_target in all_cols else 0, key="stab_target")
                    task_type = st.selectbox("Tipo de Tarefa", ["classification", "regression", "time_series", "anomaly_detection", "clustering"], key="stab_task")
                    
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
                    df_stab = None
            else:
                df_stab = None

        st.divider()

        # --- 2. PRÉ-PROCESSAMENTO & NLP ---
        st.markdown("### 2. 🔧 Pré-processamento")
        with st.expander("Configurações Avançadas de Dados", expanded=False):
            # Scaler
            scaler_type = st.selectbox("Scaler (Normalização)", ["standard", "minmax", "robust", "none"], format_func=lambda x: x.capitalize(), key="stab_scaler")
            
            # NLP Config
            st.markdown("**Processamento de Texto (NLP)**")
            use_nlp = st.checkbox("Habilitar NLP", key="stab_use_nlp")
            nlp_config = {}
            nlp_cols = []
            if use_nlp and df_stab is not None:
                text_cols = df_stab.select_dtypes(include=['object']).columns.tolist()
                if not text_cols:
                    st.warning("Nenhuma coluna de texto detectada.")
                else:
                    nlp_cols = st.multiselect("Colunas de Texto", text_cols, key="stab_nlp_cols")
                    
                    vectorizer = st.selectbox("Vetorização", ["tfidf", "count"], key="stab_vect")
                    ngram_min, ngram_max = st.slider("N-Grams Range", 1, 3, (1, 2), key="stab_ngram")
                    remove_stopwords = st.checkbox("Remover Stopwords (English)", value=True, key="stab_stop")
                    max_features = st.number_input("Max Features", 100, 10000, 1000, key="stab_max_feat")
                    
                    nlp_config = {
                        "vectorizer": vectorizer,
                        "ngram_range": (ngram_min, ngram_max),
                        "stop_words": remove_stopwords,
                        "max_features": max_features,
                        "lemmatization": st.checkbox("Lematização (WordNet - requer NLTK)", value=False, key="stab_lemma")
                    }

        st.divider()

        # --- 3. MODELO ---
        st.markdown("### 3. 🤖 Modelo")
        model_source = st.radio("Fonte do Modelo", ["Sessão Atual (AutoML)", "Model Registry", "Configuração Manual", "Upload Arquivo (.pkl/.joblib)"], key="stab_model_source")
        
        model_instance = None
        model_params = {}
        
        if model_source == "Sessão Atual (AutoML)":
            # Check for best_model
            has_best = 'best_model' in st.session_state and st.session_state['best_model'] is not None
            # Check for results list
            has_results = 'automl_results' in st.session_state and st.session_state['automl_results']
            
            model_options = []
            if has_best:
                model_options.append("Best Model")
            if has_results:
                for i, res in enumerate(st.session_state['automl_results']):
                     # res should be a dict with model_name, params, metrics
                     m_name = res.get('model_name', 'Unknown')
                     acc = res.get('accuracy', res.get('r2', 0))
                     model_options.append(f"Model {i+1}: {m_name} (Score: {acc:.4f})")
            
            if not model_options:
                st.warning("Nenhum modelo treinado nesta sessão.")
            else:
                selected_model_str = st.selectbox("Selecione o Modelo", model_options, key="stab_sess_model")
                
                if selected_model_str == "Best Model":
                     model_instance = st.session_state['best_model']
                else:
                     # Reconstruct model from params
                     try:
                         idx = int(selected_model_str.split(":")[0].replace("Model ", "")) - 1
                         res = st.session_state['automl_results'][idx]
                         m_name = res.get('model_name')
                         params = res.get('params', {})
                         
                         trainer_temp = AutoMLTrainer(task_type=task_type)
                         # Try to use _instantiate_model because params likely have prefixes from Optuna
                         # Accessing protected method is fine here
                         model_instance = trainer_temp._instantiate_model(m_name, params)
                         st.success(f"Modelo {m_name} reconstruído com sucesso.")
                     except Exception as e:
                         st.error(f"Erro ao recuperar modelo: {e}")
                
        elif model_source == "Model Registry":
            reg_models = get_registered_models()
            if reg_models:
                sel_reg_model_name = st.selectbox("Modelo Registrado", [m['name'] for m in reg_models], key="stab_reg_model")
                st.info("Para usar modelos do Registry, carregue os parâmetros manualmente abaixo ou use a sessão atual.")
            else:
                st.warning("Nenhum modelo registrado.")

        elif model_source == "Upload Arquivo (.pkl/.joblib)":
            uploaded_model = st.file_uploader("Carregar modelo treinado", type=["pkl", "joblib"], key="stab_upload")
            if uploaded_model:
                try:
                    model_instance = joblib.load(uploaded_model)
                    st.success("Modelo carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar modelo: {e}")
                
        elif model_source == "Configuração Manual":
            # Dynamic Model List from AutoMLTrainer
            trainer_stab = AutoMLTrainer(task_type=task_type)
            available_models = trainer_stab.get_supported_models()
            
            if not available_models:
                st.error(f"Nenhum modelo suportado encontrado para a tarefa: {task_type}")
            else:
                model_algo = st.selectbox("Algoritmo", available_models, key="stab_algo")
                
                st.markdown("##### Hiperparâmetros Dinâmicos")
                model_params = {}
                
                # Dynamic UI based on model selection
                if "random_forest" in model_algo or "extra_trees" in model_algo:
                    c1, c2 = st.columns(2)
                    with c1:
                        n_estimators = st.number_input("Number of Trees (n_estimators)", 10, 1000, 100, step=10, key="stab_n_est")
                        max_depth = st.number_input("Max Depth", 1, 100, 10, key="stab_max_depth")
                    with c2:
                        min_samples_split = st.number_input("Min Samples Split", 2, 20, 2, key="stab_min_samples")
                        criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"] if "classifier" in model_algo or task_type == "classification" else ["squared_error", "absolute_error"], key="stab_crit")
                    model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split, "criterion": criterion}
                
                elif "xgboost" in model_algo:
                    c1, c2 = st.columns(2)
                    with c1:
                        n_estimators = st.number_input("n_estimators", 10, 1000, 100, step=10, key="stab_xgb_n")
                        learning_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.1, format="%.3f", key="stab_xgb_lr")
                    with c2:
                        max_depth = st.number_input("Max Depth", 1, 20, 6, key="stab_xgb_d")
                        subsample = st.slider("Subsample", 0.1, 1.0, 0.8, key="stab_xgb_sub")
                    model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample}

                elif "lightgbm" in model_algo:
                    c1, c2 = st.columns(2)
                    with c1:
                        n_estimators = st.number_input("n_estimators", 10, 1000, 100, step=10, key="stab_lgb_n")
                        learning_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.1, format="%.3f", key="stab_lgb_lr")
                    with c2:
                        num_leaves = st.number_input("Num Leaves", 2, 256, 31, key="stab_lgb_leaves")
                    model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "num_leaves": num_leaves}

                elif "logistic" in model_algo:
                    c1, c2 = st.columns(2)
                    with c1:
                        C = st.number_input("Regularization (C)", 0.01, 100.0, 1.0, format="%.2f", key="stab_lr_c")
                    with c2:
                        penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], key="stab_lr_p")
                    model_params = {"C": C, "penalty": penalty if penalty != "none" else None}
                    if penalty == "elasticnet":
                        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, key="stab_lr_l1")
                        model_params["l1_ratio"] = l1_ratio
                        model_params["solver"] = "saga" # Required for elasticnet

                elif "svm" in model_algo or "svr" in model_algo:
                    c1, c2 = st.columns(2)
                    with c1:
                        C = st.number_input("C", 0.01, 100.0, 1.0, format="%.2f", key="stab_svm_c")
                        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="stab_svm_k")
                    model_params = {"C": C, "kernel": kernel}

                elif "kmeans" in model_algo:
                    n_clusters = st.number_input("Number of Clusters", 2, 50, 3, key="stab_km_k")
                    init = st.selectbox("Init", ["k-means++", "random"], key="stab_km_init")
                    model_params = {"n_clusters": n_clusters, "init": init}
                
                else:
                    # Fallback to JSON for other models
                    st.info(f"Configuração UI simplificada não disponível para {model_algo}. Use JSON abaixo.")
                    params_json = st.text_area("Parâmetros (JSON)", value="{}", height=100, key="stab_params_json")
                    try:
                        model_params = json.loads(params_json)
                    except json.JSONDecodeError:
                        model_params = {}

        st.divider()
        
        # --- 4. VALIDAÇÃO E ESTABILIDADE ---
        st.markdown("### 4. 🧪 Configuração de Teste")
        
        # 4.1 Seed Configuration
        st.markdown("**Configuração de Seed (Reprodutibilidade)**")
        seed_mode_stab = st.radio("Modo de Seed", ["Automático", "Manual"], key="stab_seed_mode", horizontal=True)
        if seed_mode_stab == "Manual":
            manual_seed_stab = st.number_input("Seed Manual", 0, 999999, 42, key="stab_manual_seed")
        else:
            manual_seed_stab = 42 # Será sobrescrito ou usado como base se necessário
            st.info("Seed será gerada automaticamente.")

        st.divider()

        # 4.2 Test Strategy
        st.markdown("**Estratégia de Estabilidade**")
        stability_type = st.radio(
            "O que você deseja testar?",
            ["Robustez a Variação de Dados (Split/CV)", 
             "Robustez à Inicialização (Seed)", 
             "Sensibilidade a Hiperparâmetros",
             "Análise Geral de Estabilidade"],
            key="stab_type"
        )
        
        # Defaults
        cv_folds = 5
        n_iterations = 10
        perturbation = 0.0
        cv_strategy = 'monte_carlo'
        test_size = 0.2
        hyperparam_name = None
        hyperparam_values = []

        if stability_type == "Análise Geral de Estabilidade":
            st.caption("Executa testes de Seed e Split para gerar um score unificado de estabilidade.")
            n_iterations = st.slider("Iterações por Teste", 5, 50, 10, key="stab_iter_gen")
            st.info("Este modo executa ambos os testes de Seed e Split (Monte Carlo) e combina os resultados.")

        elif stability_type == "Robustez a Variação de Dados (Split/CV)":
            st.caption("Testa como o modelo se comporta com diferentes divisões de treino/teste.")
            test_mode = st.radio("Método de Split", ["Simples (Holdout Repetido)", "Avançado (Cross-Validation)"], key="stab_mode_split")
        
            if test_mode == "Avançado (Cross-Validation)":
                cv_type_label = st.selectbox("Estratégia CV", ["Monte Carlo (ShuffleSplit)", "K-Fold", "Stratified K-Fold", "Time Series Split"], key="stab_cv_type")
                if cv_type_label == "K-Fold": cv_strategy = 'kfold'
                elif cv_type_label == "Stratified K-Fold": cv_strategy = 'stratified_kfold'
                elif cv_type_label == "Time Series Split": cv_strategy = 'time_series_split'
                else: cv_strategy = 'monte_carlo'
                
                n_iterations = st.slider("Número de Folds / Iterações", 2, 50, 5, key="stab_folds")
                perturbation = st.slider("Perturbação (Ruído Gaussiano)", 0.0, 0.5, 0.0, key="stab_pert")
                test_size = 0.2 # Ignored for KFold/Stratified but used for Monte Carlo
                if cv_strategy == 'monte_carlo':
                     test_size = st.slider("Tamanho Teste (Monte Carlo)", 0.1, 0.5, 0.2, key="stab_test_size_mc")
            else:
                # Simples
                cv_strategy = 'monte_carlo'
                n_iterations = st.slider("Repetições (Seeds)", 5, 50, 10, key="stab_iter_simple")
                test_size = st.slider("Tamanho do Teste", 0.1, 0.5, 0.2, key="stab_test_size")
                perturbation = st.slider("Adicionar Ruído (Perturbação)", 0.0, 0.5, 0.0, key="stab_pert_simple")

        elif stability_type == "Robustez à Inicialização (Seed)":
            st.caption("Testa se a inicialização aleatória do modelo afeta o resultado (mantendo os dados fixos).")
            n_iterations = st.slider("Número de Repetições (Seeds de Modelo)", 5, 50, 10, key="stab_iter_seed")
            test_size = st.slider("Tamanho do Split de Validação Fixa", 0.1, 0.5, 0.2, key="stab_test_size_seed")
            
        elif stability_type == "Sensibilidade a Hiperparâmetros":
            st.caption("Testa como a variação de um hiperparâmetro afeta o modelo (mantendo dados e seed fixos).")
            hyperparam_name = st.text_input("Nome do Hiperparâmetro (ex: n_estimators, C, max_depth)", value="n_estimators")
            hyperparam_values_str = st.text_input("Valores (separados por vírgula)", value="10, 50, 100, 200")
            try:
                # Tentar converter para int ou float
                vals = [v.strip() for v in hyperparam_values_str.split(',')]
                hyperparam_values = []
                for v in vals:
                    try:
                        if '.' in v: hyperparam_values.append(float(v))
                        else: hyperparam_values.append(int(v))
                    except:
                        hyperparam_values.append(v) # Keep as string if not number
            except:
                st.error("Formato de valores inválido.")
            
    with col_main:
        if st.button("🚀 Executar Análise de Estabilidade", type="primary"):
            if df_stab is not None and target_col:
                with st.spinner("Executando pipeline completo (NLP + Preprocessamento + Estabilidade)..."):
                    try:
                        # 1. Configurar Processor
                        processor = AutoMLDataProcessor(
                            task_type=task_type, 
                            target_column=target_col,
                            nlp_config=nlp_config,
                            scaler_type=scaler_type
                        )
                        
                        # 2. Executar Transformação (Passar DF completo!)
                        # nlp_cols deve ser lista de nomes ou None
                        nlp_cols_arg = nlp_cols if (use_nlp and nlp_cols) else None
                        
                        X_proc, y_proc = processor.fit_transform(df_stab, nlp_cols=nlp_cols_arg)
                        
                        # 3. Instanciar Modelo Manual (se necessário)
                        if model_source == "Configuração Manual":
                            trainer = AutoMLTrainer(task_type=task_type)
                            # model_algo holds the internal name from the dropdown
                            model_instance = trainer.create_model_instance(model_algo, model_params)
                            
                        if model_instance:
                            # 4. Executar Análise
                            analyzer = StabilityAnalyzer(model_instance, X_proc, y_proc, task_type=task_type, random_state=manual_seed_stab)
                            
                            if stability_type == "Análise Geral de Estabilidade":
                                report = analyzer.run_general_stability_check(n_iterations=n_iterations)
                                st.success("Análise Geral Concluída!")
                                
                                st.markdown("### 📊 Relatório Geral de Estabilidade")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("#### 1. Estabilidade de Inicialização (Seed)")
                                    st.caption("Variação de performance devido apenas à semente aleatória do modelo.")
                                    st.dataframe(report['seed_stability'].style.highlight_max(axis=0, color='lightgreen'))
                                with c2:
                                    st.markdown("#### 2. Estabilidade de Dados (Split)")
                                    st.caption("Variação de performance devido à divisão dos dados (Monte Carlo CV).")
                                    st.dataframe(report['split_stability'].style.highlight_max(axis=0, color='lightgreen'))
                                
                                st.markdown("---")
                                st.markdown("#### Comparativo Visual")
                                
                                # Visual Comparison for Accuracy/R2/F1
                                seed_df = report['raw_seed']
                                split_df = report['raw_split']
                                seed_df['Type'] = 'Seed Stability'
                                split_df['Type'] = 'Split Stability'
                                
                                combined_df = pd.concat([seed_df, split_df], ignore_index=True)
                                metric_cols = [c for c in combined_df.columns if c not in ['iteration', 'seed', 'split_seed', 'param_value', 'Type']]
                                
                                for metric in metric_cols:
                                    fig = px.box(combined_df, x='Type', y=metric, color='Type', title=f"Comparativo de Variância: {metric}", points="all")
                                    st.plotly_chart(fig, use_container_width=True)

                            else:
                                if stability_type == "Robustez à Inicialização (Seed)":
                                    results = analyzer.run_seed_stability(n_iterations=n_iterations)
                                elif stability_type == "Sensibilidade a Hiperparâmetros":
                                    if not hyperparam_name or not hyperparam_values:
                                        st.error("Configure os hiperparâmetros corretamente.")
                                        st.stop()
                                    results = analyzer.run_hyperparameter_stability(hyperparam_name, hyperparam_values)
                                else:
                                    # Padrão (Split/CV)
                                    results = analyzer.run_stability_test(
                                        n_iterations=n_iterations, 
                                        test_size=test_size, 
                                        perturbation=perturbation,
                                        cv_strategy=cv_strategy
                                    )
                                
                                st.success("Análise concluída com sucesso!")
                                
                                # Display Metrics
                                st.markdown("### 📊 Resultados de Estabilidade")
                                
                                # Metrics Table
                                summary_df = analyzer.calculate_stability_metrics(results)
                                st.markdown("#### Resumo e Score de Estabilidade")
                                st.dataframe(summary_df.style.highlight_max(axis=0, color='lightgreen'))
                                
                                st.markdown("---")
                                st.markdown("#### Detalhes por Iteração")
                                st.dataframe(results)
                                
                                # Visualizations
                                st.markdown("#### Visualização")
                                df_res = results
                                
                                if stability_type == "Sensibilidade a Hiperparâmetros":
                                    metric_cols = [c for c in df_res.columns if c not in ['iteration', 'param_value', 'seed', 'split_seed']]
                                    x_col = 'param_value'
                                    for metric in metric_cols:
                                        fig = px.line(df_res, x=x_col, y=metric, markers=True, title=f"{metric} vs {hyperparam_name}")
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Distribuição (Boxplot/Histograma)
                                    metric_cols = [c for c in df_res.columns if c not in ['iteration', 'seed', 'split_seed', 'param_value']]
                                    if metric_cols:
                                        c1, c2 = st.columns(2)
                                        for i, metric in enumerate(metric_cols):
                                            with (c1 if i % 2 == 0 else c2):
                                                fig = px.box(df_res, y=metric, title=f"Estabilidade: {metric}", points="all")
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                        st.markdown("#### Histograma das Métricas")
                                        c3, c4 = st.columns(2)
                                        for i, metric in enumerate(metric_cols):
                                            with (c3 if i % 2 == 0 else c4):
                                                fig = px.histogram(df_res, x=metric, nbins=15, title=f"Histograma: {metric}")
                                                st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Sem métricas numéricas para visualizar.")

                        else:
                            st.error("Modelo não configurado corretamente ou falha na criação.")
                            
                    except Exception as e:
                        st.error(f"Erro durante a análise: {e}")
                        st.exception(e)
            else:
                st.warning("👈 Por favor, selecione um dataset e configure o modelo na barra lateral esquerda.")
