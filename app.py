from automl_engine import AutoMLDataProcessor, AutoMLTrainer, save_pipeline, get_technical_explanation
from cv_engine import CVAutoMLTrainer, get_cv_explanation
import streamlit as st
import pandas as pd
import numpy as np
from mlops_utils import MLFlowTracker, DriftDetector, ModelExplainer, get_model_registry, DataLake, register_model_from_run, get_registered_models, get_all_runs
import os
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

def prepare_multi_dataset(selected_configs, global_split=None, task_type='classification', date_col=None):
    """
    Loads and splits multiple datasets based on user configurations.
    selected_configs: List of dicts with {'name': str, 'version': str, 'split': float}
    global_split: If provided (0.0 to 1.0), overrides individual split configs.
    task_type: Type of task to determine split strategy (e.g., temporal for time_series).
    date_col: Required for temporal split in time_series.
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
                # Random split for other tasks
                tr, te = train_test_split(df_ds, train_size=split_ratio, random_state=42)
            
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
    "🗂️ Model Registry"
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
    
    sub_tabs = st.tabs(["🆕 Novo Treino", "🔧 Modelos Existentes (Fine-Tune)"])
    
    # --- SUB-TAB: NOVO TREINO ---
    with sub_tabs[0]:
        st.subheader("📋 Configuração do Novo Treino")
        
        # 1. Definição da Tarefa
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            task = st.radio("Tipo de Tarefa", ["classification", "regression", "clustering", "time_series", "anomaly_detection"], key="task_selector_train")
        
        with col_t2:
            training_strategy = st.radio("Configuração de Hiperparâmetros", ["Automático", "Manual"], 
                                         help="Automático: O sistema busca os melhores parâmetros. Manual: Você define tudo.")

        st.divider()

        # 2. Configuração de Modelos e Parâmetros
        trainer_temp = AutoMLTrainer(task_type=task)
        available_models = trainer_temp.get_available_models()
        
        selected_models = None
        manual_params = None
        
        st.subheader("🎯 Configuração da Otimização")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            mode_selection = st.radio("Seleção de Modelos", ["Automático (Todos)", "Manual (Selecionar)"], horizontal=True)
            if mode_selection == "Manual (Selecionar)":
                selected_models = st.multiselect("Escolha os Modelos", available_models, default=available_models[:2])
            
            tuning_mode = st.radio("Modo de Tuning", ["Automático", "Customizado"], horizontal=True)
            if tuning_mode == "Customizado":
                n_trials = st.number_input("Número de Tentativas", 1, 500, 20)
                early_stopping = st.number_input("Early Stopping (Rounds)", 0, 50, 7)
            else:
                n_trials = 30
                early_stopping = 10
        
        with col_opt2:
            auto_split = st.checkbox("Auto-Split (Otimizar % de Treino/Validação)", value=False)
            if task == "time_series":
                st.info("💡 Split temporal obrigatório para séries temporais.")

        # Hiperparâmetros Manuais integrados nas opções de tuning
        if training_strategy == "Manual":
            st.divider()
            st.subheader("⚙️ Configuração de Hiperparâmetros Manuais")
            st.info("Nota: No modo Manual, você define os parâmetros que serão usados como ponto de partida (enqueue) para os modelos selecionados.")
            
            # Se múltiplos modelos estiverem selecionados, o usuário pode configurar um por um ou um modelo de referência
            ref_model = st.selectbox("Modelo para Configurar", selected_models or available_models)

            manual_params = {'model_name': ref_model} 
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
        
        date_col_pre = None
        if task == "time_series":
            if selected_ds_list:
                try:
                    first_ds = selected_ds_list[0]
                    versions = datalake.list_versions(first_ds)
                    if versions:
                        first_ver = versions[0]
                        sample_df = datalake.load_version(first_ds, first_ver, nrows=5)
                        date_col_pre = st.selectbox("📅 Coluna de Data (OBRIGATÓRIO)", sample_df.columns, key="ts_date_selector")
                except Exception as e: st.error(f"Erro: {e}")
            else: st.warning("Selecione um dataset.")

        selected_configs = []
        if selected_ds_list:
            cols_ds = st.columns(len(selected_ds_list))
            for i, ds_name in enumerate(selected_ds_list):
                with cols_ds[i]:
                    st.markdown(f"**{ds_name}**")
                    versions = datalake.list_versions(ds_name)
                    ver = st.selectbox(f"Versão", versions, key=f"ver_{ds_name}")
                    split = st.slider("Treino %", 0, 100, 80, key=f"split_{ds_name}")
                    selected_configs.append({'name': ds_name, 'version': ver, 'split': split})

        if selected_configs:
            if st.button("📥 Carregar e Preparar Dados", key="btn_load_train"):
                train_df, test_df = prepare_multi_dataset(selected_configs, global_split=1.0 if auto_split else None, task_type=task, date_col=date_col_pre)
                st.session_state['train_df'] = train_df
                st.session_state['test_df'] = test_df
                st.session_state['current_task'] = task
                st.session_state['date_col_active'] = date_col_pre
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
                target = st.selectbox("🎯 Target", train_df.columns) if task not in ["clustering", "anomaly_detection"] else None
            
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
                effective_models = selected_models if selected_models else trainer_temp.get_available_models()
                total_expected_trials = n_trials * len(effective_models) if training_strategy == "Automático" else 1
                
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
                        fig = px.line(df_trials, x="Trial Modelo", y="Score", color="Modelo", 
                                    markers=True, hover_name="Identificador",
                                    title="Progresso da Otimização por Algoritmo")
                        fig.update_layout(xaxis_title="Nº da Tentativa do Modelo", yaxis_title="Score (Métrica)")
                        st.plotly_chart(fig, key=f"chart_{trial.number}", use_container_width=True)

                with st.spinner("Processando..."):
                    processor = AutoMLDataProcessor(target_column=target, task_type=task, date_col=date_col_pre, forecast_horizon=forecast_horizon)
                    X_train_proc, y_train_proc = processor.fit_transform(train_df)
                    X_test_proc, y_test_proc = processor.transform(test_df) if test_df is not None else (None, None)
                    
                    trainer = AutoMLTrainer(task_type=task)
                    
                    best_model = trainer.train(
                        X_train_proc, 
                        y_train_proc, 
                        n_trials=n_trials if training_strategy == "Automático" else 1, 
                        callback=callback, 
                        selected_models=selected_models, 
                        early_stopping_rounds=early_stopping if training_strategy == "Automático" else 0,
                        manual_params=manual_params,
                        experiment_name=experiment_name
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
                                # Feature Importance (Shap simplified)
                                st.info("Calculando importância das features...")
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

    # --- SUB-TAB: MODELOS EXISTENTES (FINE-TUNE) ---
    with sub_tabs[1]:
        st.subheader("📋 Configuração de Fine-Tuning")
        
        # 1. Definição da Tarefa (Espelhado do Novo Treino)
        col_ft1, col_ft2 = st.columns(2)
        with col_ft1:
            ft_task = st.radio("Tipo de Tarefa", ["classification", "regression", "clustering", "time_series", "anomaly_detection"], key="task_selector_ft")
        
        with col_ft2:
            ft_strategy = st.radio("Estratégia de Fine-Tune", ["Automático", "Manual"], key="strategy_ft",
                                         help="Automático: O sistema busca os melhores hiperparâmetros. Manual: Você define tudo.")

        st.divider()

        # 2. Seleção da Base do Modelo (Diferença principal)
        st.subheader("🎯 Seleção do Modelo Base")
        ft_model_source = st.selectbox("Fonte do Modelo Base", 
            ["Model Registry (Registrados)", "Transformers (HuggingFace)", "Upload Local (.pkl)"])
        
        base_model_name = ""
        ft_available_models = []
        
        if ft_model_source == "Model Registry (Registrados)":
            reg_models = get_registered_models()
            if reg_models:
                base_model_name = st.selectbox("Selecione o Modelo Registrado", [m.name for m in reg_models], key="ft_reg_sel")
                ft_available_models = [base_model_name]
            else:
                st.warning("Nenhum modelo registrado encontrado.")
        
        elif ft_model_source == "Transformers (HuggingFace)":
            if ft_task == "classification":
                ft_available_models = [
                    "bert-base-uncased", 
                    "distilbert-base-uncased", 
                    "roberta-base", 
                    "albert-base-v2", 
                    "xlnet-base-cased", 
                    "microsoft/deberta-v3-base"
                ]
            elif ft_task == "regression":
                ft_available_models = [
                    "bert-base-uncased-reg", 
                    "distilbert-base-uncased-reg", 
                    "roberta-base-reg", 
                    "microsoft/deberta-v3-small"
                ]
            else:
                st.info("Transformers disponíveis principalmente para Classificação e Regressão de Texto.")
                ft_available_models = []
            
            base_model_name = st.selectbox("Selecione o Transformer", ft_available_models, key="ft_trans_sel")
        
        else: # Upload Local
            uploaded_ft_file = st.file_uploader("Upload do arquivo .pkl base", type="pkl", key="ft_upload")
            if uploaded_ft_file:
                base_model_name = "Uploaded_Model"
                ft_available_models = [base_model_name]

        # --- NOVA SEÇÃO: Parâmetros Manuais para Fine-Tune ---
        ft_manual_params = None
        if ft_strategy == "Manual" and base_model_name:
            st.divider()
            st.subheader("⚙️ Configuração Manual de Hiperparâmetros")
            st.info(f"Defina os parâmetros para o modelo base: {base_model_name}")
            
            ft_manual_params = {'model_name': base_model_name}
            # Tentar obter schema se não for Transformer ou Uploaded
            if ft_model_source == "Model Registry (Registrados)":
                # Para modelos registrados, tentamos inferir o tipo original para o schema
                # Por simplicidade, vamos usar o nome do modelo se ele bater com os conhecidos
                trainer_temp_ft = AutoMLTrainer(task_type=ft_task)
                schema_ft = trainer_temp_ft.get_model_params_schema(base_model_name)
                if schema_ft:
                    cols_p_ft = st.columns(3)
                    for i, (p_name, p_config) in enumerate(schema_ft.items()):
                        with cols_p_ft[i % 3]:
                            if p_config[0] == 'int':
                                ft_manual_params[p_name] = st.number_input(p_name, p_config[1], p_config[2], p_config[3], key=f"ft_m_{p_name}")
                            elif p_config[0] == 'float':
                                ft_manual_params[p_name] = st.number_input(p_name, p_config[1], p_config[2], p_config[3], format="%.4f", key=f"ft_m_{p_name}")
                            elif p_config[0] == 'list':
                                options, p_def = p_config[1], p_config[2]
                                ft_manual_params[p_name] = st.selectbox(p_name, options, index=options.index(p_def) if p_def in options else 0, key=f"ft_m_{p_name}")
                else:
                    st.info("Este modelo não possui parâmetros configuráveis via interface ou é um modelo customizado.")
            
            elif ft_model_source == "Transformers (HuggingFace)":
                st.markdown("**Parâmetros de Fine-Tuning para Transformer**")
                col_tr1, col_tr2 = st.columns(2)
                with col_tr1:
                    ft_manual_params['learning_rate'] = st.number_input("Learning Rate", 1e-6, 1e-3, 2e-5, format="%.6f", key="ft_tr_lr")
                with col_tr2:
                    ft_manual_params['num_train_epochs'] = st.number_input("Epochs", 1, 10, 3, key="ft_tr_epochs")
        # ---------------------------------------------------

        st.divider()

        # 3. Configuração da Otimização (Igual ao Novo Treino)
        st.subheader("⚙️ Configuração da Otimização")
        col_ft_opt1, col_ft_opt2 = st.columns(2)
        with col_ft_opt1:
            ft_tuning_mode = st.radio("Modo de Tuning", ["Automático", "Customizado"], horizontal=True, key="ft_tuning_mode")
            if ft_tuning_mode == "Customizado":
                ft_n_trials = st.number_input("Número de Tentativas", 1, 500, 10, key="ft_trials")
                ft_early_stopping = st.number_input("Early Stopping (Rounds)", 0, 50, 5, key="ft_es")
            else:
                ft_n_trials = 15
                ft_early_stopping = 5
        
        with col_ft_opt2:
            ft_auto_split = st.checkbox("Auto-Split", value=True, key="ft_auto_split")

        st.divider()

        # 4. Seleção de Dados (Igual ao Novo Treino)
        st.subheader("📂 Seleção de Dados para Retreino")
        ft_selected_ds_list = st.multiselect("Escolha os Datasets", datalake.list_datasets(), key="ds_ft_multi")
        
        ft_selected_configs = []
        if ft_selected_ds_list:
            cols_ft_ds = st.columns(len(ft_selected_ds_list))
            for i, ds_name in enumerate(ft_selected_ds_list):
                with cols_ft_ds[i]:
                    st.markdown(f"**{ds_name}**")
                    ft_versions = datalake.list_versions(ds_name)
                    ft_ver = st.selectbox(f"Versão", ft_versions, key=f"ft_ver_{ds_name}")
                    ft_split = st.slider("Treino %", 0, 100, 80, key=f"ft_split_{ds_name}")
                    ft_selected_configs.append({'name': ds_name, 'version': ft_ver, 'split': ft_split})

        if ft_selected_configs:
            if st.button("📥 Carregar Dados para Fine-Tuning", key="btn_load_ft"):
                ft_train_df, ft_test_df = prepare_multi_dataset(ft_selected_configs, global_split=1.0 if ft_auto_split else None, task_type=ft_task)
                st.session_state['ft_train_df'] = ft_train_df
                st.session_state['ft_test_df'] = ft_test_df
                st.session_state['ft_task_active'] = ft_task
                st.session_state['ft_n_trials_active'] = ft_n_trials
                st.session_state['ft_early_stopping_active'] = ft_early_stopping
                st.session_state['ft_base_model_name'] = base_model_name
                st.session_state['ft_manual_params_active'] = ft_manual_params
                st.session_state['ft_strategy_active'] = ft_strategy
                st.success("Dados carregados para Fine-Tuning!")

        if 'ft_train_df' in st.session_state and st.session_state.get('ft_task_active') == ft_task:
            ft_train_df = st.session_state['ft_train_df']
            ft_test_df = st.session_state['ft_test_df']
            ft_base_model = st.session_state['ft_base_model_name']
            
            st.divider()
            st.subheader("⚙️ Configuração Final de Fine-Tuning")
            ft_target = st.selectbox("🎯 Target (Fine-Tuning)", ft_train_df.columns, key="ft_target_sel") if ft_task not in ["clustering", "anomaly_detection"] else None
            
            if st.button("🚀 Iniciar Processo de Fine-Tuning", key="btn_exec_ft"):
                st.session_state['trials_data_ft'] = []
                ft_status_c = st.empty()
                ft_progress_bar = st.progress(0)
                ft_chart_c = st.empty()
                
                ft_strat_active = st.session_state.get('ft_strategy_active', 'Automático')
                ft_manual_p = st.session_state.get('ft_manual_params_active')
                
                # Se for manual, rodamos apenas 1 trial. Se automático, o número configurado.
                total_expected_ft = st.session_state['ft_n_trials_active'] if ft_strat_active == "Automático" else 1

                def ft_callback(trial, score, full_name, dur, metrics=None):
                    trial_info = {
                        "Tentativa Geral": trial.number + 1,
                        "Modelo": full_name.split(" - ")[0],
                        "Identificador": full_name,
                        "Score": score,
                        "Duração (s)": dur
                    }
                    if metrics:
                        for m_name, m_val in metrics.items():
                            if isinstance(m_val, (int, float, np.number)):
                                trial_info[m_name.upper()] = m_val
                    
                    st.session_state['trials_data_ft'].append(trial_info)
                    df_ft = pd.DataFrame(st.session_state['trials_data_ft'])
                    
                    with ft_status_c:
                        st.info(f"✨ Fine-Tune: {full_name} | Score: {score:.4f} | {trial.number+1}/{total_expected_ft}")
                    
                    ft_progress_bar.progress(min((trial.number + 1) / total_expected_ft, 1.0))
                    
                    with ft_chart_c:
                        fig = px.line(df_ft, x="Tentativa Geral", y="Score", title="Evolução do Fine-Tuning")
                        st.plotly_chart(fig, use_container_width=True)

                with st.spinner("Realizando Fine-Tuning..."):
                    try:
                        from automl_engine import AutoMLTrainer, AutoMLDataProcessor
                        ft_trainer = AutoMLTrainer(task_type=ft_task)
                        ft_processor = AutoMLDataProcessor(target_column=ft_target, task_type=ft_task)
                        
                        X_ft_train, y_ft_train = ft_processor.fit_transform(ft_train_df)
                        X_ft_test, y_ft_test = ft_processor.transform(ft_test_df) if ft_test_df is not None else (None, None)
                        
                        # Nome do experimento
                        ft_exp_name = f"FineTune_{ft_base_model}_{time.strftime('%Y%m%d_%H%M%S')}"
                        
                        # Executa o treino focado apenas no modelo base selecionado
                        ft_trainer.train(
                            X_ft_train, y_ft_train,
                            X_test=X_ft_test, y_test=y_ft_test,
                            n_trials=total_expected_ft,
                            selected_models=[ft_base_model],
                            callback=ft_callback,
                            early_stopping_rounds=st.session_state['ft_early_stopping_active'],
                            experiment_name=ft_exp_name,
                            manual_params=ft_manual_p
                        )
                        
                        st.success(f"Fine-Tuning de {ft_base_model} concluído!")
                        st.balloons()
                        
                        # Mostrar métricas finais
                        best_ft_score = ft_trainer.results[0]['score'] if ft_trainer.results else 0
                        st.metric("Melhor Score após Fine-Tune", f"{best_ft_score:.4f}")
                        
                    except Exception as e:
                        st.error(f"Erro no Fine-Tuning: {e}")
                
                # Use same data loading logic as AutoML for consistency
                available_ds_retrain = datalake.list_datasets()
                sel_ds_retrain = st.selectbox("Dataset para Retreino", available_ds_retrain)
                if sel_ds_retrain:
                    vers_retrain = datalake.list_versions(sel_ds_retrain)
                    sel_ver_retrain = st.selectbox("Versão do Dataset", vers_retrain)
                    df_new = datalake.load_version(sel_ds_retrain, sel_ver_retrain)
                    
                    target_retrain = st.selectbox("Selecione o Target", df_new.columns)
                    
                    if st.button("🚀 Iniciar Retreinamento"):
                        with st.spinner("Retreinando modelo..."):
                            try:
                                # Re-fit processor and model
                                X_new, y_new = loaded_processor.fit_transform(df_new)
                                loaded_model.fit(X_new, y_new)
                                
                                st.success("Modelo retreinado com sucesso!")
                                
                                # Evaluate
                                metrics_new = {}
                                y_p = loaded_model.predict(X_new)
                                if pd.api.types.is_numeric_dtype(y_new):
                                    from sklearn.metrics import r2_score, mean_absolute_error
                                    metrics_new['R2'] = r2_score(y_new, y_p)
                                    metrics_new['MAE'] = mean_absolute_error(y_new, y_p)
                                else:
                                    from sklearn.metrics import accuracy_score
                                    metrics_new['Accuracy'] = accuracy_score(y_new, y_p)
                                
                                st.write("Novas Métricas (no próprio treino):", metrics_new)
                                
                                # Save new version
                                tracker = MLFlowTracker(experiment_name="fine_tuning_retrain")
                                run_id = tracker.log_experiment(
                                    params={"retrained": True, "source": model_name_display},
                                    metrics=metrics_new,
                                    model=loaded_model,
                                    model_name=f"retrained_{model_name_display}",
                                    register=True
                                )
                                st.info(f"Nova versão registrada no MLflow! Run ID: {run_id}")
                            except Exception as e:
                                st.error(f"Erro no retreino: {e}")

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
