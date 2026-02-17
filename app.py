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
import shap
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
import mlflow

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

    st.divider()
    
    # --- DagsHub Integration ---
    with st.expander("🔗 DagsHub Integration"):
        st.caption("Conecte-se ao seu repositório DagsHub para salvar experimentos remotamente.")
        
        # Tentar recuperar configurações do ambiente (.env)
        env_user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        env_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        
        # Tentar extrair o nome do repositório da URI se for DagsHub
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
            if st.button("Conectar ao DagsHub"):
                if dh_user and dh_repo and dh_token:
                    try:
                        # Configurar variáveis de ambiente para autenticação MLflow
                        os.environ["MLFLOW_TRACKING_USERNAME"] = dh_user
                        os.environ["MLFLOW_TRACKING_PASSWORD"] = dh_token
                        
                        # Configurar URI de Tracking
                        remote_uri = f"https://dagshub.com/{dh_user}/{dh_repo}.mlflow"
                        os.environ["MLFLOW_TRACKING_URI"] = remote_uri # Atualizar env para persistência na sessão
                        mlflow.set_tracking_uri(remote_uri)
                        
                        # Tentar listar experimentos para validar conexão
                        try:
                            # Teste simples de conexão
                            mlflow.search_experiments(max_results=1)
                            #st.success(f"✅ Conectado: {dh_user}/{dh_repo}")
                            st.session_state['dagshub_connected'] = True
                            st.session_state['mlflow_uri'] = remote_uri
                        except Exception as e:
                            st.error(f"❌ Falha na conexão: {e}")
                            # Reverter para local em caso de erro
                            local_uri = "sqlite:///mlflow.db"
                            mlflow.set_tracking_uri(local_uri)
                            os.environ["MLFLOW_TRACKING_URI"] = local_uri
                    except Exception as e:
                        st.error(f"Erro ao configurar: {e}")
                else:
                    st.warning("Preencha todos os campos.")
        
        with col_dh2:
            # Botão de desconectar apenas se estiver conectado (ou se a URI apontar para DagsHub)
            is_dagshub = "dagshub.com" in mlflow.get_tracking_uri()
            if st.button("Desconectar (Voltar ao Local)", disabled=not is_dagshub):
                local_uri = "sqlite:///mlflow.db"
                mlflow.set_tracking_uri(local_uri)
                os.environ["MLFLOW_TRACKING_URI"] = local_uri
                
                # Opcional: Limpar credenciais da sessão (mas manter no env se vieram de lá?)
                # Por segurança, limpamos do os.environ para garantir desconexão real
                if "MLFLOW_TRACKING_USERNAME" in os.environ:
                    del os.environ["MLFLOW_TRACKING_USERNAME"]
                if "MLFLOW_TRACKING_PASSWORD" in os.environ:
                    del os.environ["MLFLOW_TRACKING_PASSWORD"]
                    
                st.session_state['dagshub_connected'] = False
                st.info("🔌 Desconectado. Usando MLflow local.")
                st.rerun()

        # Mostrar status atual
        current_uri = mlflow.get_tracking_uri()
        if "dagshub.com" in current_uri:
            st.success(f"🟢 Conectado ao DagsHub")
            st.caption(f"URI: {current_uri}")
        else:
            st.info("⚪ Usando MLflow Local (SQLite)")
    
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
    ensemble_config = {} # Initialize empty ensemble config

    if model_source == "AutoML Standard (Scikit-Learn/XGBoost/Transformers)":
        mode_selection = st.radio("Seleção de Modelos", ["Automático (Preset)", "Manual (Selecionar)", "Custom Ensemble Builder"], horizontal=True)
        
        if mode_selection == "Manual (Selecionar)":
            selected_models = st.multiselect("Escolha os Modelos", available_models, default=available_models[:2] if available_models else None)
            
        elif mode_selection == "Custom Ensemble Builder":
            st.markdown("##### 🏗️ Construção de Ensemble Customizado")
            st.info("Crie um ensemble combinando múltiplos modelos base. O sistema treinará o ensemble final.")
            
            ensemble_type = st.selectbox("Tipo de Ensemble", ["Voting (Votação)", "Stacking (Empilhamento)"])
            
            # Filter base models (exclude other ensembles/custom models to avoid recursion for now)
            base_candidates = [m for m in available_models if 'ensemble' not in m and 'custom' not in m]
            
            st.markdown("**1. Selecione os Estimadores Base**")
            selected_base_models = st.multiselect(
                "Estimadores Base (Componentes)", 
                base_candidates, 
                default=base_candidates[:3] if len(base_candidates) > 3 else base_candidates
            )
            
            if len(selected_base_models) < 2:
                st.warning("⚠️ Selecione pelo menos 2 modelos para formar um ensemble robusto.")
            
            if ensemble_type == "Voting (Votação)":
                st.markdown("**2. Configuração do Voting**")
                if task == "classification":
                    voting_type = st.selectbox("Tipo de Votação", ["soft", "hard"], help="Soft: Média das probabilidades. Hard: Votação majoritária das classes.")
                else:
                    voting_type = 'soft' # Not used in regressor but safe to keep
                    st.caption("Regressão usa média das predições.")
                
                use_weights = st.checkbox("Definir Pesos (Weighted Voting)", help="Permite atribuir pesos diferentes para cada modelo na votação.")
                voting_weights = None
                
                if use_weights:
                    st.caption("Insira os pesos separados por vírgula na mesma ordem dos modelos selecionados.")
                    weights_input = st.text_input("Pesos (ex: 1.0, 2.0)", value=",".join(["1.0"] * len(selected_base_models)))
                    try:
                        voting_weights = [float(w.strip()) for w in weights_input.split(',')]
                        if len(voting_weights) != len(selected_base_models):
                            st.error(f"⚠️ Número de pesos ({len(voting_weights)}) diferente do número de modelos ({len(selected_base_models)}). Usando pesos iguais.")
                            voting_weights = None
                    except:
                        st.error("⚠️ Formato inválido. Use números separados por vírgula.")
                        voting_weights = None

                ensemble_config = {
                    'voting_estimators': selected_base_models,
                    'voting_type': voting_type,
                    'voting_weights': voting_weights
                }
                selected_models = ['custom_voting']
                
            elif ensemble_type == "Stacking (Empilhamento)":
                st.markdown("**2. Configuração do Stacking**")
                st.info("Stacking treina um 'Meta-Modelo' para aprender a melhor combinação dos modelos base.")
                
                # Final estimator selection
                meta_candidates = ['logistic_regression', 'random_forest', 'xgboost', 'linear_regression', 'ridge']
                # Filter by task
                if task == 'classification':
                     meta_candidates = [m for m in meta_candidates if m in base_candidates and m != 'linear_regression' and m != 'ridge']
                     if not meta_candidates: meta_candidates = ['logistic_regression']
                else:
                     meta_candidates = [m for m in meta_candidates if m in base_candidates and m != 'logistic_regression']
                     if not meta_candidates: meta_candidates = ['linear_regression']

                final_est_name = st.selectbox("Meta-Modelo (Final Estimator)", meta_candidates)
                
                st.caption(f"Meta-Modelo selecionado: {final_est_name}")
                
                ensemble_config = {
                    'stacking_estimators': selected_base_models,
                    'stacking_final_estimator': final_est_name
                }
                selected_models = ['custom_stacking']

    
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
        # Seletor de Modo de Otimização (Novo)
        optimization_mode = st.selectbox(
            "Modo de Otimização de Hiperparâmetros",
            ["Bayesian Optimization (Padrão)", "Random Search", "Grid Search", "Hyperband"],
            index=0,
            help="Bayesian: Mais eficiente. Random: Exploratório. Grid: Exaustivo (lento). Hyperband: Rápido para muitos dados."
        )
        
        # Mapeamento para o backend
        opt_mode_map = {
            "Bayesian Optimization (Padrão)": "bayesian",
            "Random Search": "random",
            "Grid Search": "grid",
            "Hyperband": "hyperband"
        }
        selected_opt_mode = opt_mode_map[optimization_mode]

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
            total_time_budget = st.number_input("Tempo Máximo Total (segundos)", 60, 86400, 3600, key="cust_total_time", help="Tempo máximo para executar TODO o experimento. Se excedido, o treino para após o modelo atual.")
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
            total_time_budget = None
            early_stopping = 10
            manual_params = {}

        
        with col_opt2:
            st.markdown("##### 🛡️ Estratégia de Validação")
            validation_options = ["Automático (Recomendado)", "K-Fold Cross Validation", "Stratified K-Fold", "Holdout (Treino/Teste)", "Auto-Split (Otimizado)", "Time Series Split"]
            
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
            if val_strategy_ui == "Automático (Recomendado)":
                validation_strategy = 'auto'
                st.info("O sistema escolherá a melhor estratégia baseada no tamanho dos dados.")
            elif val_strategy_ui in ["K-Fold Cross Validation", "Stratified K-Fold"]:
                n_folds = st.number_input("Número de Folds", 2, 20, 5, key="val_folds")
                validation_params['folds'] = n_folds
                validation_strategy = 'cv' if val_strategy_ui == "K-Fold Cross Validation" else 'stratified_cv'
            elif val_strategy_ui == "Holdout (Treino/Teste)":
                test_size = st.slider("Tamanho do Teste (%)", 10, 50, 20, key="val_holdout", help="Porcentagem do dataset de Treino reservada para Validação Interna durante a otimização (não confundir com o Teste Final).") / 100.0
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
                        role = st.radio("Papel", ["Treino + Teste (Split)", "Apenas Treino (100%)", "Apenas Teste (100%)"], key=f"role_{ds_name}", help="Define o destino final dos dados. 'Apenas Teste' reserva os dados para avaliação final (não visto no treino). 'Treino' entra no pool de treinamento.")
                        
                        split = 100
                        if role == "Treino + Teste (Split)":
                            split = st.slider(f"% Treino", 10, 95, 80, key=f"split_{ds_name}", help="Porcentagem deste dataset que vai para o pool de Treino. O restante vai para o Teste Final.")
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
                        max_features_automl = st.number_input("Max Features", min_value=100, max_value=None, value=5000, step=1000, key="automl_max_feat", help="Deixe alto (ex: 5000+) para capturar mais vocabulário. Otimizado automaticamente.")

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

                    trainer = AutoMLTrainer(task_type=task, preset=training_preset, ensemble_config=ensemble_config)
                    
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
                        experiment_name=experiment_name,
                        random_state=random_seed_config,
                        validation_strategy=validation_strategy,
                        validation_params=validation_params,
                        custom_models=custom_models,
                        optimization_mode=selected_opt_mode
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
                                # Feature Importance (SHAP - SHapley Additive exPlanations)
                                st.markdown("#### 📈 Importância das Features (SHAP)")
                                st.info("Calculando explicabilidade via SHAP (pode levar alguns segundos)...")
                                
                                shap_success = False
                                try:
                                    # Usar sample para performance
                                    sample_train = X_train_proc
                                    if len(sample_train) > 200:
                                        sample_train = shap.utils.sample(sample_train, 200)
                                        
                                    sample_test = X_test_proc
                                    if sample_test is not None and len(sample_test) > 100:
                                        sample_test = shap.utils.sample(sample_test, 100)

                                    if sample_test is not None:
                                        explainer = ModelExplainer(best_model, sample_train, task_type=task)
                                        
                                        # Plot Beeswarm (Resumo)
                                        st.markdown("**SHAP Summary Plot**")
                                        st.caption("Mostra como cada feature impacta a saída do modelo. Pontos vermelhos = valor alto da feature, azuis = valor baixo.")
                                        fig_shap = explainer.plot_importance(sample_test, plot_type="summary")
                                        st.pyplot(fig_shap)
                                        
                                        # Plot Bar (Importância Global)
                                        st.markdown("**SHAP Feature Importance (Bar)**")
                                        st.caption("Média absoluta do impacto de cada feature.")
                                        fig_shap_bar = explainer.plot_importance(sample_test, plot_type="bar")
                                        st.pyplot(fig_shap_bar)
                                        shap_success = True
                                except Exception as e:
                                    st.warning(f"Não foi possível gerar SHAP plot: {e}")

                                # Fallback para feature importance manual se SHAP falhar
                                if not shap_success and hasattr(trainer, 'feature_importance') and trainer.feature_importance:
                                    st.info("Exibindo importância baseada em coeficientes/árvores (método alternativo).")
                                    fi_data = pd.DataFrame({
                                        'Feature': processor.get_feature_names(),
                                        'Importância': trainer.feature_importance
                                    }).sort_values(by='Importância', ascending=False)
                                    
                                    fig_fi = px.bar(fi_data.head(15), x='Importância', y='Feature', orientation='h',
                                                  title="Top 15 Features mais Importantes")
                                    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig_fi, use_container_width=True)

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
    st.header("⚖️ Análise de Estabilidade e Robustez")
    st.markdown("Avalie a confiabilidade do seu modelo sob diferentes condições.")

    col_config, col_main = st.columns([1, 2])

    with col_config:
        st.subheader("⚙️ Configuração")
        
        # --- 1. SELEÇÃO DE DADOS ---
        st.markdown("### 1. 📂 Dados")
        dataset_names = datalake.list_datasets()
        if not dataset_names:
            st.warning("Nenhum dataset no Data Lake.")
            df_stab = None
        else:
            selected_ds = st.selectbox("Dataset", dataset_names, key="stab_ds_refactored")
            if selected_ds:
                versions = datalake.list_versions(selected_ds)
                selected_ver = st.selectbox("Versão", versions, key="stab_ver_refactored")
                try:
                    df_stab = datalake.load_version(selected_ds, selected_ver)
                    st.success(f"Carregado: {len(df_stab)} linhas")
                    
                    all_cols = df_stab.columns.tolist()
                    default_target = all_cols[-1]
                    if 'target' in all_cols: default_target = 'target'
                    elif 'class' in all_cols: default_target = 'class'
                    
                    target_col = st.selectbox("Coluna Alvo (Target)", all_cols, index=all_cols.index(default_target) if default_target in all_cols else len(all_cols)-1, key="stab_target_refactored")
                    task_type = st.selectbox("Tipo de Tarefa", ["classification", "regression"], key="stab_task_refactored")
                    
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
                    df_stab = None
            else:
                df_stab = None

        st.divider()

        # --- 2. SELEÇÃO DE MODELO ---
        st.markdown("### 2. 🤖 Modelo")
        model_source = st.radio("Fonte do Modelo", ["Sessão Atual (AutoML)", "Model Registry", "Upload Arquivo (.pkl/.joblib)"], key="stab_model_source_refactored")
        
        model_instance = None
        
        if model_source == "Sessão Atual (AutoML)":
            if 'best_model' in st.session_state and st.session_state['best_model'] is not None:
                st.info(f"Usando Melhor Modelo da Sessão: {st.session_state.get('best_model_name', 'Best Model')}")
                model_instance = st.session_state['best_model']
            else:
                st.warning("Nenhum modelo treinado nesta sessão.")
                
        elif model_source == "Model Registry":
            reg_models = get_registered_models()
            if reg_models:
                sel_reg_model_name = st.selectbox("Modelo Registrado", [m['name'] for m in reg_models], key="stab_reg_model_refactored")
                try:
                    # Load model from registry (mockup logic as get_registered_models returns metadata)
                    # Assuming load_registered_model exists or we construct path
                    # Using load_registered_model from utils
                    model_instance = load_registered_model(sel_reg_model_name) 
                    if model_instance:
                        st.success(f"Modelo {sel_reg_model_name} carregado.")
                except Exception as e:
                    st.error(f"Erro ao carregar do Registry: {e}")
            else:
                st.warning("Nenhum modelo registrado.")

        elif model_source == "Upload Arquivo (.pkl/.joblib)":
            uploaded_model = st.file_uploader("Carregar modelo", type=["pkl", "joblib"], key="stab_upload_refactored")
            if uploaded_model:
                try:
                    model_instance = joblib.load(uploaded_model)
                    st.success("Modelo carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar modelo: {e}")

        st.divider()

        # --- 3. TIPO DE TESTE ---
        st.markdown("### 3. 🧪 Tipo de Teste")
        test_type = st.radio(
            "Selecione o Teste de Estabilidade:",
            [
                "Robustez a Variação de Dados",
                "Robustez à Inicialização", 
                "Sensibilidade a Hiperparâmetros", 
                "Análise Geral"
            ],
            key="stab_test_type"
        )

    with col_main:
        if df_stab is not None and model_instance is not None:
            st.subheader(f"Executando: {test_type}")
            
            # Prepare Data (Basic Preprocessing for Stability Analysis if needed)
            # Assuming model handles it or data is numeric. 
            # StabilityAnalyzer expects X, y.
            X = df_stab.drop(columns=[target_col])
            y = df_stab[target_col]
            
            # Simple encoding for categorical features if model is not a pipeline
            # If model is a pipeline, we pass raw X.
            # We try to detect if model is pipeline.
            from sklearn.pipeline import Pipeline
            is_pipeline = isinstance(model_instance, Pipeline)
            
            if not is_pipeline:
                # Basic encoding for non-pipeline models to avoid errors
                # This is a fallback. Ideally models in AutoMLOps are pipelines.
                # Only apply if we detect object columns
                obj_cols = X.select_dtypes(include=['object', 'category']).columns
                if len(obj_cols) > 0:
                    st.caption("⚠️ Modelo não é pipeline e dados contêm texto/categorias. Aplicando Ordinal Encoding simples para o teste.")
                    for c in obj_cols:
                        X[c] = X[c].astype('category').cat.codes

            # Instantiate Analyzer
            analyzer = StabilityAnalyzer(model_instance, X, y, task_type=task_type)

            # --- DYNAMIC CONFIGURATION BASED ON TEST TYPE ---
            if test_type == "Robustez a Variação de Dados":
                st.info("Testa como o desempenho varia com diferentes divisões de Treino/Teste (Split).")
                n_splits = st.slider("Número de Divisões (Splits)", 5, 50, 10, key="stab_split_n")
                test_size = st.slider("Tamanho do Teste (%)", 0.1, 0.5, 0.2, key="stab_split_size")
                
                if st.button("Executar Teste de Variação de Dados"):
                    with st.spinner("Executando..."):
                        results = analyzer.run_split_stability(n_splits=n_splits, test_size=test_size)
                        st.write("### Resultados")
                        st.dataframe(results)
                        
                        # Plot
                        if not results.empty:
                            metrics = [c for c in results.columns if c not in ['iteration', 'split_seed']]
                            for m in metrics:
                                fig = px.box(results, y=m, title=f"Variação de {m} por Split", points="all")
                                st.plotly_chart(fig, use_container_width=True)

            elif test_type == "Robustez à Inicialização":
                st.info("Testa como o desempenho varia com diferentes sementes aleatórias (Seeds) do modelo, mantendo os dados fixos.")
                n_iter = st.slider("Número de Iterações", 5, 50, 10, key="stab_seed_n")
                
                if st.button("Executar Teste de Inicialização"):
                    with st.spinner("Executando..."):
                        results = analyzer.run_seed_stability(n_iterations=n_iter)
                        st.write("### Resultados")
                        st.dataframe(results)
                        
                        if not results.empty:
                            metrics = [c for c in results.columns if c not in ['iteration', 'seed']]
                            for m in metrics:
                                fig = px.histogram(results, x=m, title=f"Distribuição de {m} (Seed Stability)", nbins=10)
                                st.plotly_chart(fig, use_container_width=True)

            elif test_type == "Sensibilidade a Hiperparâmetros":
                st.info("Testa como o desempenho varia ao alterar um hiperparâmetro específico.")
                
                # Try to guess params
                try:
                    params = model_instance.get_params()
                    param_name = st.selectbox("Selecione o Hiperparâmetro", list(params.keys()), key="stab_hp_name")
                    
                    current_val = params.get(param_name)
                    st.write(f"Valor Atual: {current_val}")
                    
                    # Manual input for values
                    values_input = st.text_input("Valores para testar (separados por vírgula)", value="0.1, 1.0, 10.0" if isinstance(current_val, (int, float)) else "gini, entropy")
                    
                    if st.button("Executar Análise de Sensibilidade"):
                        # Parse values
                        try:
                            # Try float/int first
                            if "," in values_input:
                                raw_vals = [v.strip() for v in values_input.split(",")]
                                parsed_vals = []
                                for v in raw_vals:
                                    try:
                                        if "." in v: parsed_vals.append(float(v))
                                        else: parsed_vals.append(int(v))
                                    except:
                                        parsed_vals.append(v)
                            else:
                                parsed_vals = [values_input]
                                
                            with st.spinner(f"Testando {param_name} = {parsed_vals}..."):
                                results = analyzer.run_hyperparameter_stability(param_name, parsed_vals)
                                st.write("### Resultados")
                                st.dataframe(results)
                                
                                if not results.empty:
                                    metrics = [c for c in results.columns if c not in ['param_value']]
                                    # Plot
                                    metric_to_plot = st.selectbox("Métrica para Gráfico", metrics, key="stab_hp_metric")
                                    
                                    # Check if param_value is numeric for line plot, else bar
                                    is_numeric_param = pd.to_numeric(results['param_value'], errors='coerce').notnull().all()
                                    
                                    if is_numeric_param:
                                        fig = px.line(results, x='param_value', y=metric_to_plot, title=f"Sensibilidade: {param_name} vs {metric_to_plot}", markers=True)
                                    else:
                                        fig = px.bar(results, x='param_value', y=metric_to_plot, title=f"Sensibilidade: {param_name} vs {metric_to_plot}")
                                    st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Erro ao processar valores: {e}")

                except Exception as e:
                    st.error(f"Não foi possível ler os parâmetros do modelo: {e}")

            elif test_type == "Análise Geral":
                st.info("Executa uma bateria completa de testes de estabilidade (Seed + Split) e gera um relatório resumido.")
                n_iter = st.slider("Iterações por Teste", 5, 20, 10, key="stab_general_n")
                
                if st.button("Executar Análise Geral"):
                    with st.spinner("Executando Bateria de Testes..."):
                        report = analyzer.run_general_stability_check(n_iterations=n_iter)
                        
                        st.subheader("📊 Relatório de Estabilidade")
                        
                        st.markdown("#### 1. Estabilidade de Inicialização (Seed)")
                        st.dataframe(report['seed_stability'])
                        
                        st.markdown("#### 2. Estabilidade de Dados (Split)")
                        st.dataframe(report['split_stability'])
                        
                        # Visualization of distributions
                        st.markdown("#### 3. Distribuições")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Seed Stability (Accuracy/R2)")
                            raw_seed = report['raw_seed']
                            if not raw_seed.empty:
                                main_metric = 'accuracy' if 'accuracy' in raw_seed.columns else 'r2' if 'r2' in raw_seed.columns else raw_seed.columns[0]
                                fig1 = px.box(raw_seed, y=main_metric, title=f"Seed Var: {main_metric}")
                                st.plotly_chart(fig1, use_container_width=True)
                                
                        with c2:
                            st.caption("Split Stability (Accuracy/R2)")
                            raw_split = report['raw_split']
                            if not raw_split.empty:
                                main_metric = 'accuracy' if 'accuracy' in raw_split.columns else 'r2' if 'r2' in raw_split.columns else raw_split.columns[0]
                                fig2 = px.box(raw_split, y=main_metric, title=f"Split Var: {main_metric}")
                                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("👈 Selecione um Dataset e um Modelo na barra lateral para começar.")
            if df_stab is None:
                st.warning("Dataset não selecionado.")
            if model_instance is None:
                st.warning("Modelo não selecionado.")



