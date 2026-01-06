from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from free_mlops.config import get_settings
from free_mlops.service import hash_uploaded_file
from free_mlops.service import list_experiment_records
from free_mlops.service import load_csv
from free_mlops.service import run_experiment
from free_mlops.service import save_uploaded_bytes
from free_mlops.finetune import run_finetune
from free_mlops.registry import list_registered_models
from free_mlops.registry import register_model
from free_mlops.registry import download_model_package
from free_mlops.db_delete import delete_experiment
from free_mlops.registry_delete import delete_registered_model
from free_mlops.test_models import test_single_prediction
from free_mlops.test_models import test_batch_prediction
from free_mlops.test_models import load_test_data_from_uploaded_file
from free_mlops.test_models import save_test_results
from free_mlops.monitoring import get_performance_monitor
from free_mlops.monitoring import get_alert_manager
from free_mlops.drift_detection import get_drift_detector
from free_mlops.concept_drift import get_concept_drift_detector
from free_mlops.hyperopt import get_hyperparameter_optimizer
from free_mlops.dvc_integration import get_dvc_integration
from free_mlops.data_validation import get_data_validator
from free_mlops.deep_learning import get_deep_learning_automl
from free_mlops.advanced_deep_learning import get_advanced_deep_learning_automl
from free_mlops.nlp_deep_learning import get_nlp_deep_learning_automl
from free_mlops.time_series import get_time_series_automl


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def main() -> None:
    st.set_page_config(page_title="Free MLOps", layout="wide")

    settings = get_settings()

    st.title("Free MLOps (MVP)")

    # Initialize variables to avoid UnboundLocalError
    df = None
    target_column = None
    problem_type = "classification"

    tabs = st.tabs(["Treinar", "Experimentos", "Model Registry", "Testar Modelos", "Fine-Tune", "Hyperopt", "DVC", "Data Validation", "Time Series", "Monitoramento", "Deploy/API"])

    with tabs[0]:
        st.subheader("🚀 Treinar Modelos (Clássicos + Deep Learning)")
        
        # Upload de dados
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="train_upload")
        
        if uploaded_file is not None:
            try:
                # Ler dados
                df = pd.read_csv(uploaded_file)
                st.write(f"Dataset carregado: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)
                
                # Configurações
                target_column = st.selectbox("Target (coluna alvo)", options=list(df.columns), key="train_target")
                problem_type = st.radio(
                    "Tipo do problema",
                    options=["classification", "regression"],
                    horizontal=True,
                )
                
                # Gerar ID do experimento antes de treinar
                if "experiment_id" not in st.session_state:
                    st.session_state["experiment_id"] = uuid4().hex

                st.info(f"ID do experimento: {st.session_state['experiment_id']}")

                # Sempre mostrar todos os modelos disponíveis (sem restrição de NLP)
                all_available_models = [
                    # Deep Learning
                    "mlp", "cnn", "lstm",
                    # Clássicos ensemble
                    "random_forest", "xgboost", "lightgbm",
                    # Clássicos tradicionais
                    "logistic_regression", "linear_svc", "svm", "ridge", "lasso",
                    # Outros modelos clássicos
                    "decision_tree", "knn", "extra_trees", "gradient_boosting", "elastic_net", "svr",
                    # Modelos NLP
                    "text_cnn", "text_lstm", "bert_classifier"
                ]
                
                selected_models = st.multiselect(
                    "Selecione os modelos para treinar (pode escolher múltiplos)",
                    options=all_available_models,
                    default=[],  # Nenhum modelo selecionado por padrão
                    format_func=lambda x: {
                        # Deep Learning
                        "mlp": "MLP (Rede Neural)",
                        "cnn": "CNN (Convolucional)",
                        "lstm": "LSTM (Recorrente)",
                        # Ensemble
                        "random_forest": "Random Forest",
                        "xgboost": "XGBoost",
                        "lightgbm": "LightGBM",
                        # Tradicionais
                        "logistic_regression": "Logistic Regression",
                        "linear_svc": "Linear SVC",
                        "svm": "SVM",
                        "ridge": "Ridge Regression",
                        "lasso": "Lasso Regression",
                        # Outros modelos clássicos
                        "decision_tree": "Decision Tree",
                        "knn": "KNN",
                        "extra_trees": "Extra Trees",
                        "gradient_boosting": "Gradient Boosting",
                        "elastic_net": "Elastic Net",
                        "svr": "SVR",
                        # Modelos NLP
                        "text_cnn": "Text CNN",
                        "text_lstm": "Text LSTM",
                        "bert_classifier": "BERT Classifier"
                    }[x],
                    key="selected_models",
                    help="Selecione um ou mais modelos para treinar e comparar resultados"
                )
                
                # Salvar dataset path
                dataset_path = settings.artifacts_dir / f"dataset_{st.session_state['experiment_id']}.csv"
                df.to_csv(dataset_path, index=False)
                st.session_state["dataset_path"] = str(dataset_path)
                
                # Configurações avançadas
                with st.expander("Configurações Avançadas"):
                    col1, col2 = st.columns(2)
                    with col1:
                        epochs = st.number_input("Épocas", min_value=10, max_value=1000, value=100, key="epochs")
                        batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=32, key="batch_size")
                        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, key="learning_rate")
                        max_training_time = st.number_input("Tempo Máximo (minutos)", min_value=1, max_value=120, value=30, key="max_training_time")
                    with col2:
                        dropout_rate = st.number_input("Dropout Rate", min_value=0.0, max_value=0.8, value=0.2, key="dropout_rate")
                        optimizer = st.selectbox("Optimizer", options=["adam", "sgd"], key="optimizer")
                        experiment_id = st.text_input("ID do Experimento", value=st.session_state['experiment_id'], key="experiment_id")
                    
                    # Configurações específicas para Deep Learning
                    if selected_models:
                        dl_selected_models = [m for m in selected_models if m in ["mlp", "cnn", "lstm", "text_cnn", "text_lstm", "bert_classifier"]]
                        
                        if dl_selected_models:
                            st.write("**Configurações Deep Learning:**")
                            
                            # Configurações MLP
                            if "mlp" in dl_selected_models:
                                hidden_layers_input = st.text_input(
                                    "Hidden Layers (ex: 128,64,32)", 
                                    value="128,64,32",
                                    help="Lista de neurônios em cada camada oculta, separada por vírgula",
                                    key="mlp_hidden_layers"
                                )
                                try:
                                    hidden_layers = [int(x.strip()) for x in hidden_layers_input.split(",") if x.strip().isdigit()]
                                except ValueError:
                                    hidden_layers = [128, 64, 32]
                            
                            # Configurações CNN
                            if "cnn" in dl_selected_models:
                                col1, col2 = st.columns(2)
                                with col1:
                                    conv_filters_input = st.text_input(
                                        "Conv Layers Filters (ex: 32,64)", 
                                        value="32,64",
                                        help="Número de filtros em cada camada convolucional, separado por vírgula",
                                        key="cnn_conv_filters"
                                    )
                                    try:
                                        conv_filters = [int(x.strip()) for x in conv_filters_input.split(",") if x.strip().isdigit()]
                                    except ValueError:
                                        conv_filters = [32, 64]
                                with col2:
                                    dense_layers_input = st.text_input(
                                        "Dense Layers (ex: 128,64)", 
                                        value="128,64",
                                        help="Número de neurônios em cada camada densa, separado por vírgula",
                                        key="cnn_dense_layers"
                                    )
                                    try:
                                        dense_layers = [int(x.strip()) for x in dense_layers_input.split(",") if x.strip().isdigit()]
                                    except ValueError:
                                        dense_layers = [128, 64]
                            
                            # Configurações LSTM
                            if "lstm" in dl_selected_models:
                                col1, col2 = st.columns(2)
                                with col1:
                                    hidden_dim = st.number_input("Hidden Dim", min_value=32, max_value=512, value=128, key="lstm_hidden_dim")
                                    num_layers = st.number_input("Num Layers", min_value=1, max_value=4, value=2, key="lstm_num_layers")
                                with col2:
                                    bidirectional = st.checkbox("Bidirectional LSTM", value=True, key="lstm_bidirectional")
                                    sequence_length = st.number_input("Sequence Length", min_value=5, max_value=100, value=10, key="lstm_seq_length")
                            
                            # Configurações NLP
                            if "text_cnn" in dl_selected_models or "text_lstm" in dl_selected_models:
                                col1, col2 = st.columns(2)
                                with col1:
                                    if "text_cnn" in dl_selected_models:
                                        embed_dim = st.number_input("Embedding Dim", min_value=50, max_value=300, value=128, key="text_cnn_embed_dim")
                                        num_filters = st.number_input("Num Filters", min_value=50, max_value=200, value=100, key="text_cnn_num_filters")
                                        filter_sizes = st.text_input("Filter Sizes (ex: 3,4,5)", value="3,4,5", key="text_cnn_filter_sizes")
                                        try:
                                            filter_sizes_list = [int(x.strip()) for x in filter_sizes.split(",") if x.strip().isdigit()]
                                        except ValueError:
                                            filter_sizes_list = [3, 4, 5]
                                    if "text_lstm" in dl_selected_models:
                                        embed_dim = st.number_input("Embedding Dim", min_value=50, max_value=300, value=128, key="text_lstm_embed_dim")
                                        hidden_dim = st.number_input("Hidden Dim", min_value=32, max_value=256, value=64, key="text_lstm_hidden_dim")
                                        num_layers = st.number_input("Num Layers", min_value=1, max_value=4, value=2, key="text_lstm_num_layers")
                                        bidirectional = st.checkbox("Bidirectional LSTM", value=True, key="text_lstm_bidirectional")
                                with col2:
                                    max_features = st.number_input("Max Features", min_value=1000, max_value=50000, value=10000, key="max_features")
                                    max_length = st.number_input("Max Sequence Length", min_value=50, max_value=500, value=256, key="max_length")
                        
                        # Valores padrão para variáveis não definidas
                        if 'hidden_layers' not in locals():
                            hidden_layers = [128, 64, 32]
                        if 'conv_filters' not in locals():
                            conv_filters = [32, 64]
                        if 'dense_layers' not in locals():
                            dense_layers = [128, 64]
                        if 'hidden_dim' not in locals():
                            hidden_dim = 128
                        if 'num_layers' not in locals():
                            num_layers = 2
                        if 'bidirectional' not in locals():
                            bidirectional = True
                        if 'sequence_length' not in locals():
                            sequence_length = 10
                        if 'embed_dim' not in locals():
                            embed_dim = 128
                        if 'num_filters' not in locals():
                            num_filters = 100
                        if 'filter_sizes_list' not in locals():
                            filter_sizes_list = [3, 4, 5]
                        if 'max_features' not in locals():
                            max_features = 10000
                        if 'max_length' not in locals():
                            max_length = 256
                    
                    # Configurações avançadas (gerais para todos os modelos)
                
                # Botão de treinamento unificado
                if st.button("Treinar Modelo", type="primary"):
                    with st.spinner("Treinando modelo..."):
                        try:
                            # Verificar se há modelos selecionados
                            if not selected_models:
                                st.error("Por favor, selecione pelo menos um modelo para treinar.")
                                return
                            
                            # Preparar dados
                            df_clean = df.dropna(subset=[target_column]).reset_index(drop=True)
                            feature_cols = [c for c in df_clean.columns if c != target_column]
                            X = df_clean[feature_cols]
                            y = df_clean[target_column]
                            
                            # Tratamento robusto de dados para Deep Learning
                            # Converter para numérico e tratar NaN/infinitos
                            import numpy as np
                            
                            # Converter features para numérico (se necessário)
                            X_processed = X.copy()
                            for col in X_processed.columns:
                                if X_processed[col].dtype == 'object':
                                    # Tentar converter colunas categóricas para numérico
                                    try:
                                        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                                    except:
                                        # Se não conseguir, usar label encoding
                                        X_processed[col] = X_processed[col].astype('category').cat.codes
                                else:
                                    # Para colunas numéricas, tratar valores problemáticos
                                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                            
                            # Tratar valores NaN/infinitos
                            X_processed = X_processed.fillna(X_processed.mean()).fillna(0)
                            X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Converter target para numérico se necessário
                            if problem_type == "classification":
                                try:
                                    y_processed = pd.to_numeric(y, errors='coerce')
                                    if y_processed.isna().any():
                                        # Se houver NaN no target, usar label encoding
                                        y_processed = y.astype('category').cat.codes
                                    else:
                                        y_processed = y_processed.fillna(0).astype(int)
                                except:
                                    y_processed = y.astype('category').cat.codes
                            else:
                                y_processed = pd.to_numeric(y, errors='coerce').fillna(y.mean())
                            
                            # Usar dados processados
                            X = X_processed
                            y = y_processed
                            
                            # Detectar se é Deep Learning ou clássico
                            deep_learning_models = ["mlp", "cnn", "lstm", "text_cnn", "text_lstm", "bert_classifier"]
                            
                            # Treinar todos os modelos selecionados (unificado)
                            all_results = []
                            
                            for model_name in selected_models:
                                try:
                                    from free_mlops.unified_deep_learning import get_unified_deep_learning_automl
                                except ImportError as e:
                                    st.error(f"Erro ao importar módulo Deep Learning unificado: {str(e)}")
                                    st.error("Verifique se as bibliotecas PyTorch e TensorFlow estão instaladas.")
                                    return
                                
                                # Gerar ID único para este modelo específico
                                model_experiment_id = f"{st.session_state['experiment_id']}_{model_name}_{uuid4().hex[:8]}"
                                
                                try:
                                    # Configuração personalizada completa
                                    custom_config = {
                                        "epochs": epochs,
                                        "batch_size": batch_size,
                                        "learning_rate": learning_rate,
                                        "dropout_rate": dropout_rate,
                                        "optimizer": optimizer,
                                        "activation": "relu",
                                        "max_training_time": max_training_time,
                                        "experiment_id": model_experiment_id,
                                    }
                                    
                                    # Adicionar configurações específicas
                                    if model_name == "mlp":
                                        custom_config["hidden_layers"] = hidden_layers
                                    elif model_name == "cnn":
                                        custom_config["conv_filters"] = conv_filters
                                        custom_config["dense_layers"] = dense_layers
                                    elif model_name == "lstm":
                                        custom_config["hidden_dim"] = hidden_dim
                                        custom_config["num_layers"] = num_layers
                                        custom_config["bidirectional"] = bidirectional
                                        custom_config["sequence_length"] = sequence_length
                                    elif model_name in ["text_cnn", "text_lstm"]:
                                        custom_config["max_features"] = max_features
                                        custom_config["max_length"] = max_length
                                        if model_name == "text_cnn":
                                            custom_config["embed_dim"] = embed_dim
                                            custom_config["num_filters"] = num_filters
                                            custom_config["filter_sizes"] = filter_sizes_list
                                        else:  # text_lstm
                                            custom_config["embed_dim"] = embed_dim
                                            custom_config["hidden_dim"] = hidden_dim
                                            custom_config["num_layers"] = num_layers
                                            custom_config["bidirectional"] = bidirectional
                                    
                                    # Verificar se é modelo Deep Learning ou clássico
                                    deep_learning_models = ["mlp", "cnn", "lstm", "text_cnn", "text_lstm", "bert_classifier", "tab_transformer"]
                                    
                                    if model_name in deep_learning_models:
                                        # Usar módulo unificado para todos os modelos DL
                                        automl = get_unified_deep_learning_automl()
                                        framework = "pytorch"
                                        
                                        # Preparar dados baseado no tipo de modelo
                                        if model_name in ["text_cnn", "text_lstm", "bert_classifier"]:
                                            # Modelos NLP - converter para texto
                                            if hasattr(X, 'values'):
                                                if len(X.columns) == 1 and X.iloc[:, 0].dtype == 'object':
                                                    texts = X.iloc[:, 0].astype(str).tolist()
                                                else:
                                                    texts = X.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
                                            else:
                                                texts = X.astype(str).tolist()
                                            
                                            labels = y.astype(str).tolist()
                                            
                                            # Split dados
                                            from sklearn.model_selection import train_test_split
                                            texts_train, texts_val, y_train, y_val = train_test_split(
                                                texts, labels, test_size=0.2, random_state=42
                                            )
                                            
                                            # Treinar modelo NLP
                                            result = automl.create_model(
                                                texts_train, y_train, texts_val, y_val,
                                                model_type=model_name,
                                                framework=framework,
                                                problem_type=problem_type,
                                                config=custom_config
                                            )
                                        else:
                                            # Modelos DL tradicionais - dados tabulares
                                            from sklearn.model_selection import train_test_split
                                            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                                            
                                            # Treinar modelo DL tradicional
                                            result = automl.create_model(
                                                X_train, y_train, X_val, y_val,
                                                model_type=model_name,
                                                framework=framework,
                                                problem_type=problem_type,
                                                config=custom_config
                                            )
                                    else:
                                        # Modelos clássicos - usar AutoML tradicional
                                        max_time_seconds = max_training_time * 60  # Converter para segundos
                                        
                                        record = run_experiment(
                                            dataset_path=Path(st.session_state["dataset_path"]),
                                            target_column=target_column,
                                            problem_type=problem_type,
                                            settings=settings,
                                            selected_models=[model_name],  # Apenas este modelo
                                            max_time_seconds=max_time_seconds,
                                        )
                                        all_results.append(record)
                                        continue
                                    
                                except Exception as model_error:
                                    st.error(f"Erro ao treinar modelo {model_name}: {str(model_error)}")
                                    continue
                                
                                # Salvar experimento no banco de dados (unificado para todos os DL)
                                if model_name in deep_learning_models:
                                    # Adaptar resultado para o formato esperado pelo run_deep_learning_experiment
                                    adapted_result = {
                                        "model": result.get("model"),
                                        "metrics": result.get("metrics", {}),
                                        "training_time": result.get("training_time", 0),
                                        "config": result.get("config", {}),
                                        "validation_metrics": result.get("metrics", {}),  # Chave esperada
                                        "input_shape": result.get("input_shape"),
                                        "num_classes": result.get("num_classes"),
                                        "history": result.get("history", {}),
                                    }
                                    
                                    from free_mlops.service import run_deep_learning_experiment
                                    experiment_record = run_deep_learning_experiment(
                                        dataset_path=Path(st.session_state["dataset_path"]),
                                        target_column=target_column,
                                        problem_type=problem_type,
                                        settings=settings,
                                        model_type=model_name,
                                        config=custom_config,
                                        result=adapted_result
                                    )
                                else:
                                    # Modelos clássicos - já foi salvo acima
                                    continue
                                
                                all_results.append(experiment_record)
                                
                                st.success(f"Modelos treinados! {len(all_results)} modelos concluidos")
                                
                                # Salvar todos os resultados para exibição
                                if all_results:
                                    st.session_state["all_results"] = all_results
                                    st.session_state["last_experiment"] = all_results[-1]
                                    
                        except Exception as exc:
                            st.error(f"Erro no treinamento: {str(exc)}")
                
                # Mostrar resultados de todos os modelos treinados
                all_results = st.session_state.get("all_results", [])
                
                if all_results:
                    st.subheader("Resultados do Treinamento")
                    
                    # Mostrar cada modelo individualmente
                    for i, record in enumerate(all_results):
                        with st.expander(f"📊 {record.get('best_model_name', 'Desconhecido').replace('_', ' ').title()}", expanded=i==len(all_results)-1):
                                # Informações básicas
                                col1, col2, col3 = st.columns(3)
                                best_model_name = record.get("best_model_name", "Desconhecido")
                                framework = record.get("model_metadata", {}).get("framework", "pytorch")
                                training_time = record.get('model_metadata', {}).get('training_time_seconds', 0)
                                
                                col1.metric("Modelo", best_model_name.replace("_", " ").title())
                                col2.metric("Framework", framework)
                                col3.metric("Tempo de Treino", f"{training_time:.2f}s")
                                
                                # Métricas
                                st.write("**Métricas de Validação:**")
                                metrics = record.get("best_metrics", {})
                                
                                if problem_type == "classification":
                                    col1, col2, col3, col4 = st.columns(4)
                                    val_loss = metrics.get('val_loss', 0)
                                    val_accuracy = metrics.get('val_accuracy', 0)
                                    
                                    col1.metric("Val Loss", f"{val_loss:.4f}")
                                    col2.metric("Val Accuracy", f"{val_accuracy:.4f}")
                                    
                                    if 'precision' in metrics:
                                        col3.metric("Precision", f"{metrics['precision']:.4f}")
                                    else:
                                        col3.metric("Precision", "N/A")
                                    
                                    if 'recall' in metrics:
                                        col4.metric("Recall", f"{metrics['recall']:.4f}")
                                    else:
                                        col4.metric("Recall", "N/A")
                                    
                                    if 'f1_score' in metrics:
                                        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                                    elif 'f1' in metrics:
                                        st.metric("F1-Score", f"{metrics['f1']:.4f}")
                                    else:
                                        st.metric("F1-Score", "N/A")
                                    
                                    # Matriz de Confusão
                                    if 'confusion_matrix' in metrics and metrics['confusion_matrix']:
                                        st.write("**Matriz de Confusão:**")
                                        cm = metrics['confusion_matrix']
                                        
                                        import plotly.graph_objects as go
                                        fig = go.Figure(data=go.Heatmap(
                                            z=cm,
                                            x=[f'Pred {i}' for i in range(len(cm))],
                                            y=[f'True {i}' for i in range(len(cm))],
                                            colorscale='Blues',
                                            text=cm,
                                            texttemplate="%{text}",
                                            textfont={"size": 12}
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"Matriz de Confusão - {best_model_name.replace('_', ' ').title()}",
                                            xaxis_title="Predito",
                                            yaxis_title="Verdadeiro"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True, key=f"classic_confusion_matrix_{best_model_name}_{i}")
                                    
                                    # Métricas por classe (se disponível)
                                    if 'classification_report' in metrics:
                                        st.write("**Métricas por Classe:**")
                                        report = metrics['classification_report']
                                        class_names = report.get('class_names', [])
                                        
                                        if class_names:
                                            # Criar tabela de métricas por classe
                                            class_metrics_data = []
                                            for class_name in class_names:
                                                class_metrics_data.append({
                                                    'Classe': class_name,
                                                    'Precision': report['precision'].get(class_name, 0),
                                                    'Recall': report['recall'].get(class_name, 0),
                                                    'F1-Score': report['f1_score'].get(class_name, 0),
                                                    'Support': report['support'].get(class_name, 0)
                                                })
                                            
                                            class_df = pd.DataFrame(class_metrics_data)
                                            st.dataframe(class_df, use_container_width=True)
                                
                                elif problem_type == "regression":
                                    col1, col2, col3, col4 = st.columns(4)
                                    val_loss = metrics.get('val_loss', 0)
                                    mae = metrics.get('mae', 0)
                                    mse = metrics.get('mse', 0)
                                    rmse = metrics.get('rmse', 0)
                                    
                                    col1.metric("Val Loss", f"{val_loss:.4f}")
                                    col2.metric("MAE", f"{mae:.4f}")
                                    col3.metric("MSE", f"{mse:.4f}")
                                    col4.metric("RMSE", f"{rmse:.4f}")
                                
                                # Informações do experimento
                                st.write("**Informações do Experimento:**")
                                col1, col2 = st.columns(2)
                                col1.write(f"**ID:** {record.get('id', 'N/A')}")
                                col1.write(f"**Dataset:** {record.get('dataset_path', 'N/A')}")
                                col1.write(f"**Target:** {record.get('target_column', 'N/A')}")
                                col2.write(f"**Tipo:** {record.get('problem_type', 'N/A')}")
                                col2.write(f"**Linhas:** {record.get('n_rows', 'N/A')}")
                                col2.write(f"**Colunas:** {record.get('n_cols', 'N/A')}")
                    
                    # Resultados dos modelos clássicos (se houver)
                    classic_records = [r for r in all_results if r.get("model_metadata", {}).get("framework") == "scikit-learn"]
                    if classic_records:
                        st.write("### 📊 Modelos e Métricas")
                        
                        with st.expander("📊 Resultados Completos", expanded=True):
                            # Informações básicas
                            col1, col2, col3 = st.columns(3)
                            classic_record = classic_records[0]  # Pegar o primeiro registro clássico
                            best_model_name = classic_record.get("best_model_name", "Desconhecido")
                            framework = classic_record.get("model_metadata", {}).get("framework", "scikit-learn")
                            training_time = classic_record.get('model_metadata', {}).get('training_time_seconds', 0)
                            
                            col1.metric("Melhor Modelo", best_model_name.replace("_", " ").title())
                            col2.metric("Framework", framework)
                            col3.metric("Tempo de Treino", f"{training_time:.2f}s")
                            
                            # Leaderboard
                            if 'leaderboard' in classic_record and classic_record['leaderboard']:
                                st.write("**Leaderboard Completo:**")
                                leaderboard_df = pd.DataFrame(classic_record['leaderboard'])
                                st.dataframe(leaderboard_df, use_container_width=True)
                            
                            # Métricas do melhor modelo
                            st.write("**Métricas do Melhor Modelo:**")
                            metrics = classic_record.get("best_metrics", {})
                            
                            if problem_type == "classification":
                                col1, col2, col3, col4 = st.columns(4)
                                accuracy = metrics.get('accuracy', 0)
                                f1_weighted = metrics.get('f1_weighted', 0)
                                
                                col1.metric("Accuracy", f"{accuracy:.4f}")
                                col2.metric("F1-Score (Weighted)", f"{f1_weighted:.4f}")
                                
                                if 'precision' in metrics:
                                    col3.metric("Precision", f"{metrics['precision']:.4f}")
                                else:
                                    col3.metric("Precision", "N/A")
                                
                                if 'recall' in metrics:
                                    col4.metric("Recall", f"{metrics['recall']:.4f}")
                                else:
                                    col4.metric("Recall", "N/A")
                                
                                # F1-Score geral
                                if 'f1_score' in metrics:
                                    st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                                elif 'f1' in metrics:
                                    st.metric("F1-Score", f"{metrics['f1']:.4f}")
                                
                                # Matriz de Confusão
                                if 'confusion_matrix' in metrics and metrics['confusion_matrix']:
                                    st.write("**Matriz de Confusão:**")
                                    cm = metrics['confusion_matrix']
                                    
                                    import plotly.graph_objects as go
                                    fig = go.Figure(data=go.Heatmap(
                                        z=cm,
                                        x=[f'Pred {i}' for i in range(len(cm))],
                                        y=[f'True {i}' for i in range(len(cm))],
                                        colorscale='Blues',
                                        text=cm,
                                        texttemplate="%{text}",
                                        textfont={"size": 12}
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"Matriz de Confusão - {best_model_name.replace('_', ' ').title()}",
                                        xaxis_title="Predito",
                                        yaxis_title="Verdadeiro"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"dl_confusion_matrix_{best_model_name}_{i}")
                                
                                # Métricas por classe (se disponível)
                                if 'classification_report' in metrics:
                                    st.write("**Métricas por Classe:**")
                                    report = metrics['classification_report']
                                    class_names = report.get('class_names', [])
                                    
                                    if class_names:
                                        # Criar tabela de métricas por classe
                                        class_metrics_data = []
                                        for class_name in class_names:
                                            class_metrics_data.append({
                                                'Classe': class_name,
                                                'Precision': report['precision'].get(class_name, 0),
                                                'Recall': report['recall'].get(class_name, 0),
                                                'F1-Score': report['f1_score'].get(class_name, 0),
                                                'Support': report['support'].get(class_name, 0)
                                            })
                                        
                                        class_df = pd.DataFrame(class_metrics_data)
                                        st.dataframe(class_df, use_container_width=True)
                            
                            elif problem_type == "regression":
                                col1, col2, col3, col4 = st.columns(4)
                                mae = metrics.get('mae', 0)
                                mse = metrics.get('mse', 0)
                                rmse = metrics.get('rmse', 0)
                                r2 = metrics.get('r2', 0)
                                
                                col1.metric("MAE", f"{mae:.4f}")
                                col2.metric("MSE", f"{mse:.4f}")
                                col3.metric("RMSE", f"{rmse:.4f}")
                                col4.metric("R²", f"{r2:.4f}")
                
                # Botão para baixar modelo (se houver experimento)
                if record:
                    model_path = Path(record.get("model_path", ""))
                    if model_path.exists():
                        st.download_button(
                            label="Baixar Modelo",
                            data=_read_bytes(model_path),
                            file_name=f"model_{record['id']}.pkl",
                            mime="application/octet-stream",
                            key=f"download_{record['id']}",
                        )
                    
                    # Botão para registrar modelo
                    if st.button("Registrar Modelo no Registry", key=f"register_{record['id']}"):
                        try:
                            new_record = register_model(
                                settings=settings,
                                experiment_id=record["id"],
                                new_version="v1.1.0",
                                description="Modelo treinado via UI",
                            )
                            st.success(f"Modelo registrado: {new_record['id']}")
                        except Exception as exc:
                            st.error(f"Erro ao registrar modelo: {exc}")

            except Exception as exc:
                st.error(f"Erro ao ler dataset: {exc}")

    with tabs[1]:
        st.subheader("Experimentos")

        try:
            experiments = list_experiment_records(settings=settings, limit=200, offset=0)
        except Exception as exc:
            st.error(f"Falha ao carregar experimentos: {exc}")
            experiments = []

        if not experiments:
            st.info("Nenhum experimento encontrado.")
        else:
            df_exp = pd.DataFrame(
                [
                    {
                        "id": e["id"],
                        "created_at": e["created_at"],
                        "problem_type": e["problem_type"],
                        "target_column": e["target_column"],
                        "best_model_name": e["best_model_name"],
                        "training_time_seconds": e.get("model_metadata", {}).get("training_time_seconds", 0),
                    }
                    for e in experiments
                ]
            )
            st.dataframe(df_exp, use_container_width=True)

            selected_id = st.selectbox("Ver detalhes", options=[e["id"] for e in experiments], key="exp_details")
            selected = next((e for e in experiments if e["id"] == selected_id), None)
            if selected:
                st.write("Métricas")
                st.json(selected.get("best_metrics", {}))

                st.write("Leaderboard")
                st.dataframe(pd.DataFrame(selected.get("leaderboard", [])), use_container_width=True)

                model_path = Path(selected["model_path"])
                if model_path.exists():
                    st.download_button(
                        label="Baixar modelo (.pkl)",
                        data=_read_bytes(model_path),
                        file_name=f"model_{selected['id']}.pkl",
                        mime="application/octet-stream",
                        key=f"download_selected_{selected['id']}",
                    )

                # Botão para excluir experimento
                if st.button("Excluir experimento", key=f"delete_{selected['id']}"):
                    try:
                        delete_experiment(settings.db_path, selected['id'])
                        st.success(f"Experimento {selected['id']} excluído")
                        st.experimental_rerun()
                    except Exception as exc:
                        st.error(f"Erro ao excluir experimento: {exc}")

    with tabs[2]:
        st.subheader("Model Registry")

        try:
            registered = list_registered_models(settings=settings, limit=200, offset=0)
        except Exception as exc:
            st.error(f"Falha ao carregar modelos registrados: {exc}")
            registered = []

        if not registered:
            st.info("Nenhum modelo registrado ainda.")
        else:
            df_reg = pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "version": r["model_version"],
                        "model_name": r["best_model_name"],
                        "problem_type": r["problem_type"],
                        "target": r["target_column"],
                        "created_at": r["created_at"],
                        "training_time_seconds": r.get("model_metadata", {}).get("training_time_seconds", 0),
                    }
                    for r in registered
                ]
            )
            st.dataframe(df_reg, use_container_width=True)

            selected_reg_id = st.selectbox(
                "Ver detalhes / registrar nova versão",
                options=[r["id"] for r in registered],
                key="reg_select",
            )
            selected_reg = next((r for r in registered if r["id"] == selected_reg_id), None)
            if selected_reg:
                st.write("Metadados")
                st.json(selected_reg.get("model_metadata", {}))

                st.write("Registrar nova versão")
                with st.form("register_form"):
                    new_version = st.text_input("Nova versão (ex: v1.1.0)", value="v1.1.0")
                    description = st.text_area("Descrição (opcional)", value="")
                    submitted = st.form_submit_button("Registrar versão")
                    if submitted:
                        try:
                            new_record = register_model(
                                settings=settings,
                                experiment_id=selected_reg_id,
                                new_version=new_version,
                                description=description,
                            )
                            st.success(f"Nova versão registrada: {new_record['id']}")
                        except Exception as exc:
                            st.error(str(exc))

                model_path = Path(selected_reg["model_path"])
                if model_path.exists():
                    st.download_button(
                        label="Baixar modelo (.pkl)",
                        data=_read_bytes(model_path),
                        file_name=f"model_{selected_reg['id']}.pkl",
                        mime="application/octet-stream",
                        key=f"download_reg_{selected_reg['id']}",
                    )

                # Botão para excluir modelo registrado
                if st.button("Excluir modelo registrado", key=f"delete_reg_{selected_reg['id']}"):
                    try:
                        delete_registered_model(settings.db_path, selected_reg['id'])
                        st.success(f"Modelo registrado {selected_reg['id']} excluído")
                        st.experimental_rerun()
                    except Exception as exc:
                        st.error(f"Erro ao excluir modelo registrado: {exc}")

    with tabs[3]:
        st.subheader("Testar Modelos")

        # Carregar todos os modelos disponíveis
        try:
            experiments = list_experiment_records(settings=settings, limit=200, offset=0)
            registered = list_registered_models(settings=settings, limit=200, offset=0)
        except Exception as exc:
            st.error(f"Falha ao carregar modelos: {exc}")
            experiments = []
            registered = []

        all_models = experiments + registered
        if not all_models:
            st.info("Nenhum modelo encontrado para testar.")
        else:
            # Selecionar modelo
            model_options = {
                f"{m['id']} – {m['best_model_name']} ({'Registrado' if m in registered else 'Experimento'})": m 
                for m in all_models
            }
            selected_label = st.selectbox("Escolha um modelo para testar", options=list(model_options.keys()), key="test_model_select")
            selected_model = model_options[selected_label]

            model_path = Path(selected_model["model_path"])
            if not model_path.exists():
                st.error(f"Modelo não encontrado: {model_path}")
            else:
                st.info(f"Modelo carregado: {selected_model['best_model_name']}")

                # Tabs para teste único vs em lote
                test_tabs = st.tabs(["Teste Único", "Teste em Lote"])

                with test_tabs[0]:
                    st.subheader("Teste Único")
                    
                    # Obter feature columns do metadados
                    feature_columns = selected_model.get("model_metadata", {}).get("feature_columns", [])
                    
                    if not feature_columns:
                        st.warning("Não foi possível determinar as colunas de features do modelo.")
                    else:
                        st.write("Preencha os valores para cada feature:")
                        input_data = {}
                        
                        for col in feature_columns:
                            # Tentar inferir tipo da coluna
                            input_data[col] = st.text_input(f"{col}", value="0")
                        
                        if st.button("Realizar Predição Única", type="primary"):
                            try:
                                # Converter strings para números quando possível
                                processed_input = {}
                                for k, v in input_data.items():
                                    try:
                                        if '.' in v:
                                            processed_input[k] = float(v)
                                        else:
                                            processed_input[k] = int(v)
                                    except ValueError:
                                        processed_input[k] = v
                                
                                result = test_single_prediction(model_path, processed_input)
                                
                                if result["success"]:
                                    st.success("Predição realizada com sucesso!")
                                    col1, col2 = st.columns(2)
                                    col1.metric("Predição", result["prediction"])
                                    if result["probabilities"]:
                                        col2.write("Probabilidades:")
                                        col2.json(result["probabilities"])
                                    
                                    # Botão para salvar resultado
                                    if st.button("Salvar Resultado", key="save_single"):
                                        output_path = settings.artifacts_dir / f"test_single_{selected_model['id']}.json"
                                        save_test_results(result, output_path)
                                        st.success(f"Resultado salvo em: {output_path}")
                                else:
                                    st.error(f"Erro na predição: {result['error']}")
                            except Exception as exc:
                                st.error(f"Erro ao processar: {exc}")

                with test_tabs[1]:
                    st.subheader("Teste em Lote")
                    
                    # Upload de CSV para teste em lote
                    batch_file = st.file_uploader("Upload CSV para teste em lote", type=["csv"])
                    
                    if batch_file is not None:
                        try:
                            # Ler CSV do upload
                            batch_data = load_test_data_from_uploaded_file(
                                batch_file, 
                                sample_size=st.number_input("Tamanho da amostra (0 = todos)", min_value=0, value=10, step=1) or None
                            )
                            
                            st.write(f"Carregados {len(batch_data)} registros para teste")
                            st.dataframe(pd.DataFrame(batch_data).head(), use_container_width=True)
                            
                            if st.button("Realizar Predição em Lote", type="primary"):
                                with st.spinner("Processando lote..."):
                                    result = test_batch_prediction(model_path, batch_data)
                                    
                                    if result["success"]:
                                        st.success(f"Predições realizadas para {result['batch_size']} registros!")
                                        
                                        # Mostrar resultados
                                        results_df = pd.DataFrame({
                                            "input": [str(record) for record in result["input_data"]],
                                            "prediction": result["predictions"]
                                        })
                                        
                                        if result["probabilities"]:
                                            results_df["probabilities"] = result["probabilities"]
                                        
                                        st.dataframe(results_df, use_container_width=True)
                                        
                                        # Botão para salvar resultados
                                        if st.button("Salvar Resultados em Lote", key="save_batch"):
                                            output_path = settings.artifacts_dir / f"test_batch_{selected_model['id']}.json"
                                            save_test_results(result, output_path)
                                            st.success(f"Resultados salvos em: {output_path}")
                                    else:
                                        st.error(f"Erro na predição em lote: {result['error']}")
                        except Exception as exc:
                            st.error(f"Erro ao processar arquivo: {exc}")

    with tabs[4]:
        st.subheader("Fine-Tune (ajuste de hiperparâmetros)")

        try:
            experiments = list_experiment_records(settings=settings, limit=200, offset=0)
        except Exception as exc:
            st.error(f"Falha ao carregar experimentos: {exc}")
            experiments = []

        if not experiments:
            st.info("Nenhum experimento encontrado para fine-tune.")
        else:
            exp_options = {f"{e['id']} – {e['best_model_name']}": e for e in experiments}
            selected_label = st.selectbox("Escolha um experimento para fine-tune", options=list(exp_options.keys()), key="finetune_exp_select")
            selected_exp = exp_options[selected_label]

            st.write("Detalhes do experimento base")
            st.json(selected_exp["best_metrics"])

            model_type = st.selectbox(
                "Modelo base para fine-tune",
                options=[
                    # Modelos Transformers pré-treinados
                    "bert_classification", "bert_regression", "distilbert", "roberta",
                    # Modelos clássicos para fine-tune
                    "logistic_regression", "linear_svc", "decision_tree", "knn", "extra_trees", "gradient_boosting", "elastic_net", "svr"
                ],
                format_func=lambda x: {
                    # Transformers
                    "bert_classification": "🤖 BERT (Classificação)",
                    "bert_regression": "🤖 BERT (Regressão)", 
                    "distilbert": "🚀 DistilBERT",
                    "roberta": "⚡ RoBERTa",
                    # Clássicos
                    "logistic_regression": "📈 Logistic Regression",
                    "linear_svc": "📊 Linear SVC",
                    "random_forest": "🌲 Random Forest",
                    "extra_trees": "🌳 Extra Trees",
                    "gradient_boosting": "📈 Gradient Boosting",
                    "decision_tree": "🌳 Decision Tree",
                    "knn": "🎯 KNN",
                    "ridge": "📐 Ridge",
                    "lasso": "🎯 Lasso",
                    "elastic_net": "🔗 Elastic Net",
                    "svr": "📊 SVR"
                }[x],
                index=0,
                key="finetune_model_select",
            )

            search_type = st.radio("Tipo de busca", options=["grid", "random"], horizontal=True)

            if st.button("Iniciar Fine-Tune", type="primary"):
                with st.spinner("Rodando fine-tune..."):
                    try:
                        from free_mlops.automl import ProblemType
                        from sklearn.model_selection import train_test_split

                        df = load_csv(Path(selected_exp["dataset_path"]))
                        target = selected_exp["target_column"]
                        problem_type = ProblemType(selected_exp["problem_type"])

                        df = df.dropna(subset=[target]).reset_index(drop=True)
                        feature_cols = [c for c in df.columns if c != target]
                        X = df[feature_cols]
                        y = df[target]

                        if problem_type == "regression":
                            y = pd.to_numeric(y, errors="raise")

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        result = run_finetune(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            problem_type=problem_type,
                            model_name=model_name,
                            search_type=search_type,
                            random_state=42,
                            cv_folds=3,
                        )

                        st.success("Fine-tune concluído")
                        st.json(
                            {
                                "model_name": result["model_name"],
                                "search_type": result["search_type"],
                                "best_params": result["best_params"],
                                "best_score": result["best_score"],
                                "metrics": result["metrics"],
                            }
                        )
                        
                        # Botão para excluir resultado do fine-tune (apenas da UI)
                        if st.button("Limpar resultado do fine-tune", key="clear_finetune"):
                            st.experimental_rerun()
                    except Exception as exc:
                        st.error(f"Erro no fine-tune: {exc}")

    with tabs[5]:
        st.subheader("Hyperparameter Optimization (Optuna)")
        
        # Upload de dados para otimização
        uploaded_opt = st.file_uploader("Dataset para otimização (CSV)", type=["csv"])
        
        if uploaded_opt is not None:
            try:
                # Ler dados
                df = pd.read_csv(uploaded_opt)
                st.write(f"Dataset carregado: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)
                
                # Selecionar target e tipo
                target_column = st.selectbox("Target (coluna alvo)", options=list(df.columns), key="train_target")
                problem_type = st.radio(
                    "Tipo do problema",
                    options=["classification", "regression", "multiclass_classification", "binary_classification"],
                    horizontal=True,
                )
                
                # Selecionar modelo
                model_options = [
                    "logistic_regression",
                    "ridge", 
                    "lasso",
                    "elastic_net",
                    "random_forest_classifier",
                    "random_forest_regressor", 
                    "gradient_boosting_classifier",
                    "gradient_boosting_regressor",
                    "decision_tree_classifier",
                    "decision_tree_regressor",
                    "knn_classifier",
                    "knn_regressor",
                    "svr",
                ]
                
                # Filtrar modelos por tipo de problema
                if problem_type in ["classification", "multiclass_classification", "binary_classification"]:
                    available_models = [m for m in model_options if "classifier" in m or m in ["logistic_regression"]]
                else:
                    available_models = [m for m in model_options if "regressor" in m or m in ["ridge", "lasso", "elastic_net", "svr"]]
                
                selected_model = st.selectbox("Modelo para otimizar", options=available_models, key="opt_model_select")
                
                # Configurações da otimização
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_trials = st.number_input("Número de trials", min_value=10, max_value=1000, value=100, key="opt_n_trials")
                with col2:
                    cv_folds = st.number_input("CV Folds", min_value=2, max_value=10, value=5, key="opt_cv_folds")
                with col3:
                    timeout = st.number_input("Timeout (segundos, 0 = ilimitado)", min_value=0, value=300, step=60, key="opt_timeout")
                
                timeout_seconds = None if timeout == 0 else timeout
                
                # Botão para iniciar otimização
                if st.button("Iniciar Otimização", type="primary"):
                    with st.spinner("Otimizando hiperparâmetros..."):
                        try:
                            # Preparar dados
                            df_clean = df.dropna(subset=[target_column]).reset_index(drop=True)
                            feature_cols = [c for c in df_clean.columns if c != target_column]
                            X = df_clean[feature_cols]
                            y = df_clean[target_column]
                            
                            # Split treino/validação
                            from sklearn.model_selection import train_test_split
                            X_train, X_val, y_train, y_val = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Otimizar
                            optimizer = get_hyperparameter_optimizer()
                            result = optimizer.optimize_hyperparameters(
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val,
                                model_name=selected_model,
                                problem_type=problem_type,
                                n_trials=n_trials,
                                timeout=timeout_seconds,
                                cv_folds=cv_folds,
                                random_state=42,
                            )
                            
                            st.success("✅ Otimização concluída!")
                            
                            # Mostrar resultados
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Best Score", f"{result['best_score']:.4f}")
                            col2.metric("Trials", result['n_trials'])
                            col3.metric("Modelo", result['model_name'])
                            
                            st.write("**Melhores Hiperparâmetros:**")
                            st.json(result['best_params'])
                            
                            st.write("**Métricas de Validação:**")
                            st.json(result['validation_metrics'])
                            
                            # Salvar modelo otimizado
                            if st.button("Salvar Modelo Otimizado", key=f"save_hyperopt_{result['study_name']}"):
                                try:
                                    # Salvar pipeline
                                    model_path = settings.artifacts_dir / f"hyperopt_{result['study_name']}.pkl"
                                    import joblib
                                    joblib.dump(result['best_pipeline'], model_path)
                                    
                                    st.success(f"Modelo salvo em: {model_path}")
                                except Exception as exc:
                                    st.error(f"Erro ao salvar modelo: {exc}")
                            
                        except ImportError:
                            st.error("Optuna não está instalado. Instale com: pip install optuna")
                        except Exception as exc:
                            st.error(f"Erro na otimização: {exc}")
            
            except Exception as exc:
                st.error(f"Erro ao carregar dataset: {exc}")
        
        # Histórico de otimizações
        st.write("**Histórico de Otimizações:**")
        try:
            optimizer = get_hyperparameter_optimizer()
            studies = optimizer.list_studies()
            
            if studies:
                studies_df = pd.DataFrame(studies)
                st.dataframe(studies_df, use_container_width=True)
                
                # Detalhes de estudo selecionado
                selected_study = st.selectbox(
                    "Ver detalhes do estudo",
                    options=[s["study_name"] for s in studies],
                    key="study_select"
                )
                
                if selected_study:
                    study_data = next((s for s in studies if s["study_name"] == selected_study), None)
                    if study_data:
                        st.write("**Detalhes do Estudo:**")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Modelo", study_data["model_name"])
                        col2.metric("Tipo", study_data["problem_type"])
                        col3.metric("Best Score", f"{study_data['best_score']:.4f}")
                        
                        # Histórico de trials
                        history = optimizer.get_optimization_history(selected_study)
                        if history:
                            st.write("**Histórico de Trials:**")
                            history_df = pd.DataFrame([
                                {
                                    "Trial": t["number"],
                                    "Score": t["value"],
                                    "Estado": t["state"],
                                    "Parâmetros": str(t["params"])[:100] + "..." if len(str(t["params"])) > 100 else str(t["params"]),
                                }
                                for t in history
                            ])
                            st.dataframe(history_df, use_container_width=True)
            else:
                st.info("Nenhuma otimização realizada ainda.")
                
        except Exception as exc:
            st.error(f"Erro ao carregar histórico: {exc}")

    with tabs[6]:
        st.subheader("DVC - Data Version Control")
        
        dvc = get_dvc_integration()
        
        # Status do DVC
        st.write("**Status do DVC:**")
        dvc_status = dvc.get_dvc_status()
        
        if not dvc_status.get("initialized", False):
            st.warning("DVC não está inicializado. Use o botão abaixo para inicializar.")
            
            if st.button("Inicializar DVC", type="primary"):
                with st.spinner("Inicializando DVC..."):
                    try:
                        result = dvc.initialize_dvc()
                        if result.get("initialized", False):
                            st.success("✅ DVC inicializado com sucesso!")
                            for msg in result.get("messages", []):
                                st.info(msg)
                            st.experimental_rerun()
                        else:
                            st.error("❌ Erro ao inicializar DVC")
                            for msg in result.get("messages", []):
                                st.error(msg)
                    except Exception as exc:
                        st.error(f"Erro: {exc}")
        else:
            st.success("✅ DVC está inicializado e pronto para uso")
            
            # Tabs DVC
            dvc_tabs = st.tabs(["Datasets", "Pipelines", "Linhagem"])
            
            with dvc_tabs[0]:
                st.subheader("Gerenciamento de Datasets")
                
                # Upload de novo dataset
                st.write("**Adicionar Novo Dataset:**")
                dataset_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
                
                if dataset_file is not None:
                    with st.form("add_dataset_form"):
                        dataset_name = st.text_input("Nome do Dataset", value=dataset_file.name.replace('.csv', ''))
                        dataset_type = st.selectbox("Tipo", options=["raw", "processed", "models"], key="dataset_type_select")
                        description = st.text_area("Descrição (opcional)")
                        tags_input = st.text_input("Tags (separadas por vírgula)")
                        
                        submitted = st.form_submit_button("Adicionar Dataset ao DVC")
                        
                        if submitted:
                            try:
                                # Salvar arquivo temporariamente
                                temp_path = settings.data_dir / f"temp_{dataset_name}.csv"
                                temp_path.write_bytes(dataset_file.getvalue())
                                
                                tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
                                
                                result = dvc.add_dataset(
                                    dataset_path=temp_path,
                                    dataset_name=dataset_name,
                                    dataset_type=dataset_type,
                                    description=description,
                                    tags=tags,
                                    metadata={"original_filename": dataset_file.name}
                                )
                                
                                if result["success"]:
                                    st.success(f"✅ Dataset {dataset_name} adicionado ao DVC!")
                                    st.info(result["message"])
                                else:
                                    st.error("❌ Erro ao adicionar dataset")
                                
                                # Limpar arquivo temporário
                                temp_path.unlink(missing_ok=True)
                                
                            except Exception as exc:
                                st.error(f"Erro: {exc}")
                                # Limpar arquivo temporário
                                temp_path.unlink(missing_ok=True)
                
                # Listar datasets
                st.write("**Datasets Existentes:**")
                try:
                    datasets = dvc.list_datasets()
                    
                    if datasets:
                        # Filtros
                        col1, col2 = st.columns(2)
                        with col1:
                            filter_type = st.selectbox("Filtrar por tipo", options=["todos", "raw", "processed", "models"], key="filter_type_select")
                        with col2:
                            filter_tags = st.text_input("Filtrar por tags")
                        
                        # Aplicar filtros
                        filtered_datasets = datasets
                        if filter_type != "todos":
                            filtered_datasets = [d for d in filtered_datasets if d.get("type") == filter_type]
                        if filter_tags:
                            filter_tag_list = [tag.strip() for tag in filter_tags.split(",")]
                            filtered_datasets = [d for d in filtered_datasets if any(tag in d.get("tags", []) for tag in filter_tag_list)]
                        
                        if filtered_datasets:
                            datasets_df = pd.DataFrame([
                                {
                                    "Nome": d["name"],
                                    "Tipo": d["type"],
                                    "Tamanho (bytes)": d["file_size"],
                                    "Criado em": d["created_at"][:19],
                                    "Tags": ", ".join(d.get("tags", [])),
                                    "Descrição": d.get("description", "")[:50] + "..." if len(d.get("description", "")) > 50 else d.get("description", ""),
                                }
                                for d in filtered_datasets
                            ])
                            st.dataframe(datasets_df, use_container_width=True)
                            
                            # Detalhes do dataset selecionado
                            selected_dataset = st.selectbox(
                                "Ver detalhes",
                                options=[d["name"] for d in filtered_datasets],
                                key="dataset_select"
                            )
                            
                            if selected_dataset:
                                dataset_info = dvc.get_dataset_info(selected_dataset)
                                if dataset_info:
                                    st.write("**Detalhes do Dataset:**")
                                    st.json(dataset_info)
                                    
                                    # Versões
                                    versions = dvc.get_dataset_versions(selected_dataset)
                                    if versions:
                                        st.write("**Versões:**")
                                        versions_df = pd.DataFrame(versions)
                                        st.dataframe(versions_df, use_container_width=True)
                                        
                                        # Checkout de versão
                                        selected_version = st.selectbox(
                                            "Fazer checkout de versão",
                                            options=[v["hash"] for v in versions],
                                            format_func=lambda x: f"{x[:8]} - {v.get('date', '')}",
                                            key="version_select"
                                        )
                                        
                                        if st.button("Fazer Checkout", key=f"checkout_{selected_dataset}"):
                                            try:
                                                result = dvc.checkout_dataset_version(selected_dataset, selected_version)
                                                if result["success"]:
                                                    st.success(f"✅ {result['message']}")
                                                    st.info(f"Arquivo restaurado em: {result['target_path']}")
                                                else:
                                                    st.error("❌ Erro no checkout")
                                            except Exception as exc:
                                                st.error(f"Erro: {exc}")
                        else:
                            st.info("Nenhum dataset encontrado com os filtros selecionados.")
                    else:
                        st.info("Nenhum dataset encontrado. Adicione um dataset usando o formulário acima.")
                        
                except Exception as exc:
                    st.error(f"Erro ao listar datasets: {exc}")
            
            with dvc_tabs[1]:
                st.subheader("Pipelines DVC")
                
                # Criar novo pipeline
                st.write("**Criar Pipeline:**")
                with st.form("create_pipeline_form"):
                    pipeline_name = st.text_input("Nome do Pipeline")
                    pipeline_desc = st.text_area("Descrição (opcional)")
                    
                    st.write("**Estágios do Pipeline:**")
                    stages = []
                    stage_count = st.number_input("Número de estágios", min_value=1, max_value=10, value=1)
                    
                    for i in range(stage_count):
                        with st.expander(f"Estágio {i+1}"):
                            stage_name = st.text_input(f"Nome do estágio {i+1}", value=f"stage_{i+1}", key=f"stage_name_{i}")
                            stage_cmd = st.text_input(f"Comando", value=f"echo 'Stage {i+1}'", key=f"stage_cmd_{i}")
                            stage_deps = st.text_input(f"Dependências (separadas por espaço)", key=f"stage_deps_{i}")
                            stage_outs = st.text_input(f"Outputs (separados por espaço)", key=f"stage_outs_{i}")
                            
                            stages.append({
                                "name": stage_name,
                                "cmd": stage_cmd,
                                "deps": stage_deps.split() if stage_deps else [],
                                "outs": stage_outs.split() if stage_outs else [],
                            })
                    
                    submitted = st.form_submit_button("Criar Pipeline")
                    
                    if submitted:
                        try:
                            result = dvc.create_pipeline(pipeline_name, stages, pipeline_desc)
                            if result["success"]:
                                st.success(f"✅ Pipeline {pipeline_name} criado!")
                                st.info(result["message"])
                            else:
                                st.error("❌ Erro ao criar pipeline")
                        except Exception as exc:
                            st.error(f"Erro: {exc}")
                
                # Executar pipeline
                st.write("**Executar Pipeline:**")
                if st.button("Executar Pipeline Atual", type="primary"):
                    with st.spinner("Executando pipeline..."):
                        try:
                            result = dvc.run_pipeline("current")
                            if result["success"]:
                                st.success("✅ Pipeline executado com sucesso!")
                                if result.get("stdout"):
                                    st.code(result["stdout"])
                            else:
                                st.error("❌ Erro na execução do pipeline")
                                if result.get("stderr"):
                                    st.error(result["stderr"])
                        except Exception as exc:
                            st.error(f"Erro: {exc}")
            
            with dvc_tabs[2]:
                st.subheader("Linhagem de Dados")
                
                # Selecionar dataset para análise
                try:
                    datasets = dvc.list_datasets()
                    if datasets:
                        selected_lineage_dataset = st.selectbox(
                            "Selecionar Dataset para Análise de Linhagem",
                            options=[d["name"] for d in datasets],
                            key="lineage_dataset_select"
                        )
                        
                        if selected_lineage_dataset:
                            with st.spinner("Analisando linhagem..."):
                                lineage = dvc.get_data_lineage(selected_lineage_dataset)
                                
                                if "error" in lineage:
                                    st.error(f"Erro na análise: {lineage['error']}")
                                else:
                                    st.write("**Linhagem de Dados:**")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Dataset", lineage["dataset"])
                                        st.metric("Dependências", len(lineage.get("dependencies", [])))
                                    with col2:
                                        st.metric("Downstream", len(lineage.get("downstream", [])))
                                    
                                    # Dependências
                                    if lineage.get("dependencies"):
                                        st.write("**Dependências (Datasets de entrada):**")
                                        for dep in lineage["dependencies"]:
                                            st.write(f"- {dep}")
                                    
                                    # Downstream
                                    if lineage.get("downstream"):
                                        st.write("**Downstream (Datasets gerados):**")
                                        for down in lineage["downstream"]:
                                            st.write(f"- {down}")
                                    
                                    # Grafo (se disponível)
                                    if lineage.get("graph"):
                                        st.write("**Grafo de Dependências:**")
                                        st.code(lineage["graph"])
                    else:
                        st.info("Nenhum dataset encontrado. Adicione datasets primeiro.")
                        
                except Exception as exc:
                    st.error(f"Erro na análise de linhagem: {exc}")

    with tabs[7]:
        st.subheader("Data Validation (Pandera)")
        
        validator = get_data_validator()
        
        # Tabs de validação
        val_tabs = st.tabs(["Criar Schema", "Validar Dados", "Schemas", "Comparação"])
        
        with val_tabs[0]:
            st.subheader("Criar Schema de Validação")
            
            # Upload de dados para criar schema
            schema_file = st.file_uploader("Upload Dataset para criar Schema (CSV)", type=["csv"])
            
            if schema_file is not None:
                try:
                    # Ler dados
                    df = pd.read_csv(schema_file)
                    st.write(f"Dataset carregado: {df.shape}")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Configurações do schema
                    with st.form("create_schema_form"):
                        schema_name = st.text_input("Nome do Schema", value=schema_file.name.replace('.csv', '_schema'))
                        description = st.text_area("Descrição (opcional)")
                        strict = st.checkbox("Strict (valida colunas extras)", value=True)
                        coerce = st.checkbox("Coerce (converte tipos)", value=True)
                        
                        submitted = st.form_submit_button("Criar Schema")
                        
                        if submitted:
                            with st.spinner("Criando schema..."):
                                try:
                                    result = validator.create_schema_from_dataframe(
                                        df=df,
                                        schema_name=schema_name,
                                        description=description,
                                        strict=strict,
                                        coerce=coerce,
                                    )
                                    
                                    st.success("✅ Schema criado com sucesso!")
                                    st.info(f"Schema salvo como: {result['schema_name']}")
                                    
                                    # Mostrar informações do schema
                                    schema_info = result["schema_info"]
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Colunas", len(schema_info["columns"]))
                                    col2.metric("Strict", schema_info["strict"])
                                    col3.metric("Coerce", schema_info["coerce"])
                                    
                                    # Detalhes das colunas
                                    st.write("**Análise das Colunas:**")
                                    columns_df = pd.DataFrame([
                                        {
                                            "Coluna": col,
                                            "Tipo": info["pandas_dtype"],
                                            "Nullable": info["nullable"],
                                            "Únicos": info["unique_count"],
                                            "Total": info["total_count"],
                                        }
                                        for col, info in schema_info["columns"].items()
                                    ])
                                    st.dataframe(columns_df, use_container_width=True)
                                    
                                except ImportError:
                                    st.error("Pandera não está instalado. Instale com: pip install pandera")
                                except Exception as exc:
                                    st.error(f"Erro ao criar schema: {exc}")
                
                except Exception as exc:
                    st.error(f"Erro ao ler dataset: {exc}")
        
        with val_tabs[1]:
            st.subheader("Validar Dados")
            
            # Upload de dados para validar
            validation_file = st.file_uploader("Upload Dataset para Validar (CSV)", type=["csv"])
            
            if validation_file is not None:
                try:
                    # Ler dados
                    df = pd.read_csv(validation_file)
                    st.write(f"Dataset carregado: {df.shape}")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Selecionar schema
                    schemas = validator.list_schemas()
                    if schemas:
                        schema_options = {s["name"]: s for s in schemas}
                        selected_schema = st.selectbox(
                            "Schema para validação",
                            options=list(schema_options.keys()),
                            key="validation_schema_select"
                        )
                        
                        if selected_schema:
                            # Mostrar info do schema
                            schema_info = schema_options[selected_schema]
                            col1, col2 = st.columns(2)
                            col1.metric("Schema", selected_schema)
                            col2.metric("Colunas", schema_info["columns_count"])
                            
                            lazy_validation = st.checkbox("Validação Lazy (mostra todos os erros)")
                            
                            if st.button("Validar Dataset", type="primary"):
                                with st.spinner("Validando dados..."):
                                    try:
                                        result = validator.validate_dataframe(
                                            df=df,
                                            schema_name=selected_schema,
                                            lazy=lazy_validation,
                                        )
                                        
                                        if result["success"]:
                                            st.success("✅ Dataset validado com sucesso!")
                                            col1, col2 = st.columns(2)
                                            col1.metric("Shape Original", result["original_shape"])
                                            col2.metric("Shape Validado", result["validated_shape"])
                                        else:
                                            st.error("❌ Falha na validação!")
                                            
                                            # Mostrar erros
                                            if result["errors"]:
                                                st.write("**Erros encontrados:**")
                                                errors_df = pd.DataFrame(result["errors"])
                                                st.dataframe(errors_df, use_container_width=True)
                                            
                                            col1.metric("Shape", result["original_shape"])
                                            col2.metric("Erros", len(result["errors"]))
                                        
                                        # Timestamp da validação
                                        st.info(f"Validação realizada em: {result['validation_time'][:19]}")
                                        
                                    except Exception as exc:
                                        st.error(f"Erro na validação: {exc}")
                    else:
                        st.warning("Nenhum schema encontrado. Crie um schema primeiro.")
                
                except Exception as exc:
                    st.error(f"Erro ao ler dataset: {exc}")
        
        with val_tabs[2]:
            st.subheader("Schemas Disponíveis")
            
            schemas = validator.list_schemas()
            
            if schemas:
                # Tabela de schemas
                schemas_df = pd.DataFrame(schemas)
                st.dataframe(schemas_df, use_container_width=True)
                
                # Detalhes do schema selecionado
                selected_details_schema = st.selectbox(
                    "Ver detalhes do schema",
                    options=[s["name"] for s in schemas],
                    key="details_schema_select"
                )
                
                if selected_details_schema:
                    schema_details = validator.get_schema_details(selected_details_schema)
                    if schema_details:
                        st.write("**Detalhes do Schema:**")
                        
                        # Metadados
                        metadata = schema_details.get("metadata", {})
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Nome", metadata.get("name", ""))
                        col2.metric("Criado em", metadata.get("created_at", "")[:19])
                        col3.metric("Strict", metadata.get("strict", False))
                        
                        # Colunas
                        st.write("**Colunas:**")
                        columns_data = metadata.get("columns", {})
                        if columns_data:
                            columns_details_df = pd.DataFrame([
                                {
                                    "Coluna": col,
                                    "Tipo": info["pandas_dtype"],
                                    "Nullable": info["nullable"],
                                    "Únicos": info["unique_count"],
                                    "Total": info["total_count"],
                                }
                                for col, info in columns_data.items()
                            ])
                            st.dataframe(columns_details_df, use_container_width=True)
                        
                        # Histórico de validações
                        st.write("**Histórico de Validações:**")
                        validation_history = validator.get_validation_history(selected_details_schema)
                        
                        if validation_history:
                            history_df = pd.DataFrame([
                                {
                                    "Data": result["validation_time"][:19],
                                    "Sucesso": result["success"],
                                    "Shape": str(result["original_shape"]),
                                    "Erros": len(result.get("errors", [])),
                                }
                                for result in validation_history[:10]  # Últimas 10
                            ])
                            st.dataframe(history_df, use_container_width=True)
                        else:
                            st.info("Nenhuma validação realizada ainda.")
                        
                        # Exportar schema
                        st.write("**Exportar Schema:**")
                        export_format = st.selectbox("Formato", options=["json", "python"], key="export_format_select")
                        
                        if st.button("Exportar Schema", key=f"export_{selected_details_schema}"):
                            try:
                                exported = validator.export_schema(selected_details_schema, export_format)
                                st.download_button(
                                    label=f"Baixar Schema ({export_format})",
                                    data=exported,
                                    file_name=f"{selected_details_schema}.{export_format}",
                                    mime="text/plain",
                                )
                            except Exception as exc:
                                st.error(f"Erro ao exportar: {exc}")
            else:
                st.info("Nenhum schema encontrado. Crie um schema primeiro.")
        
        with val_tabs[3]:
            st.subheader("Comparação de Schemas")
            
            schemas = validator.list_schemas()
            
            if len(schemas) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    schema1 = st.selectbox("Schema 1", options=[s["name"] for s in schemas], key="schema1_select")
                with col2:
                    schema2 = st.selectbox("Schema 2", options=[s["name"] for s in schemas], key="schema2_select")
                
                if schema1 and schema2 and schema1 != schema2:
                    if st.button("Comparar Schemas", type="primary"):
                        with st.spinner("Comparando schemas..."):
                            try:
                                comparison = validator.compare_schemas(schema1, schema2)
                                
                                if "error" in comparison:
                                    st.error(comparison["error"])
                                else:
                                    if comparison["identical"]:
                                        st.success("✅ Os schemas são idênticos!")
                                    else:
                                        st.warning("⚠️ Os schemas apresentam diferenças:")
                                    
                                    # Colunas adicionadas
                                    if comparison["columns_added"]:
                                        st.write("**Colunas Adicionadas:**")
                                        for col in comparison["columns_added"]:
                                            st.write(f"- + {col}")
                                    
                                    # Colunas removidas
                                    if comparison["columns_removed"]:
                                        st.write("**Colunas Removidas:**")
                                        for col in comparison["columns_removed"]:
                                            st.write(f"- - {col}")
                                    
                                    # Colunas modificadas
                                    if comparison["columns_modified"]:
                                        st.write("**Colunas Modificadas:**")
                                        for mod in comparison["columns_modified"]:
                                            st.write(f"- ~ {mod['column']}")
                                            with st.expander(f"Detalhes de {mod['column']}"):
                                                col1, col2 = st.columns(2)
                                                col1.write("**Antes:**")
                                                col1.json(mod["old"])
                                                col2.write("**Depois:**")
                                                col2.json(mod["new"])
                                
                            except Exception as exc:
                                st.error(f"Erro na comparação: {exc}")
            else:
                st.info("É necessário ter pelo menos 2 schemas para comparar.")

    with tabs[8]:
        st.subheader("Time Series (ARIMA, Prophet, LSTM)")
        
        ts_automl = get_time_series_automl()
        
        # Tabs de Time Series
        ts_tabs = st.tabs(["Treinar Modelo", "Modelos Salvos", "Previsões", "Análise"])
        
        with ts_tabs[0]:
            st.subheader("Treinar Modelo de Time Series")
            
            # Upload de dados
            ts_file = st.file_uploader("Upload Dataset Time Series (CSV)", type=["csv"])
            
            if ts_file is not None:
                try:
                    # Ler dados
                    df = pd.read_csv(ts_file)
                    st.write(f"Dataset carregado: {df.shape}")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Selecionar colunas
                    col1, col2 = st.columns(2)
                    with col1:
                        date_column = st.selectbox("Coluna de Data", options=list(df.columns), key="ts_date_col")
                    with col2:
                        value_column = st.selectbox("Coluna de Valor", options=[c for c in df.columns if c != date_column], key="ts_value_col")
                    
                    # Tipo de modelo
                    model_type = st.selectbox(
                        "Tipo de Modelo",
                        options=["arima", "prophet", "lstm"],
                        format_func=lambda x: {
                            "arima": "📊 ARIMA",
                            "prophet": "🔮 Prophet", 
                            "lstm": "🔄 LSTM"
                        }[x],
                        key="ts_model_type"
                    )
                    
                    # Configurações específicas por modelo
                    with st.expander("Configurações Avançadas"):
                        if model_type == "arima":
                            st.write("**Configurações ARIMA:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                p = st.number_input("AR (p)", min_value=0, max_value=5, value=1, key="arima_p")
                                d = st.number_input("I (d)", min_value=0, max_value=2, value=1, key="arima_d")
                            with col2:
                                q = st.number_input("MA (q)", min_value=0, max_value=5, value=1, key="arima_q")
                                seasonal = st.checkbox("Seasonal ARIMA", value=False, key="arima_seasonal")
                        elif model_type == "prophet":
                            st.write("**Configurações Prophet:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                yearly_seasonality = st.checkbox("Sazonalidade Anual", value=True, key="prophet_yearly")
                                weekly_seasonality = st.checkbox("Sazonalidade Semanal", value=False, key="prophet_weekly")
                            with col2:
                                daily_seasonality = st.checkbox("Sazonalidade Diária", value=False, key="prophet_daily")
                                changepoint_prior_scale = st.number_input("Changepoint Prior Scale", min_value=0.01, max_value=1.0, value=0.05, format="%.2f", key="prophet_changepoint")
                        elif model_type == "lstm":
                            st.write("**Configurações LSTM:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                sequence_length = st.number_input("Sequence Length", min_value=5, max_value=50, value=10, key="lstm_seq_len")
                                hidden_units = st.number_input("Hidden Units", min_value=32, max_value=256, value=64, key="lstm_hidden")
                            with col2:
                                num_layers = st.number_input("Num Layers", min_value=1, max_value=4, value=2, key="lstm_layers")
                                dropout_rate = st.number_input("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, key="lstm_dropout")
                    
                    # Test size
                    test_size = st.number_input("Período de Teste (dias)", min_value=7, max_value=365, value=30)
                    
                    if st.button("Treinar Modelo Time Series", type="primary"):
                        with st.spinner("Preparando dados e treinando modelo..."):
                            try:
                                # Preparar dados
                                ts_df = df[[date_column, value_column]].copy()
                                ts_df[date_column] = pd.to_datetime(ts_df[date_column])
                                ts_df = ts_df.sort_values(date_column)
                                ts_df = ts_df.set_index(date_column)
                                ts_series = ts_df[value_column].asfreq('D').dropna()
                                
                                if len(ts_series) < test_size + 30:
                                    st.error("Dados insuficientes para treinamento.")
                                    return
                                
                                # Configuração do modelo
                                config = {}
                                if model_type == "arima":
                                    config = {
                                        "p": p, "d": d, "q": q,
                                        "seasonal": seasonal
                                    }
                                elif model_type == "prophet":
                                    config = {
                                        "yearly_seasonality": yearly_seasonality,
                                        "weekly_seasonality": weekly_seasonality,
                                        "daily_seasonality": daily_seasonality,
                                        "changepoint_prior_scale": changepoint_prior_scale
                                    }
                                elif model_type == "lstm":
                                    config = {
                                        "sequence_length": sequence_length,
                                        "hidden_units": hidden_units,
                                        "num_layers": num_layers,
                                        "dropout_rate": dropout_rate,
                                        "epochs": 100,
                                        "batch_size": 32
                                    }
                                
                                # Treinar modelo
                                result = ts_automl.create_model(
                                    ts_series, model_type, test_size, config
                                )
                                
                                st.success("✅ Modelo Time Series treinado com sucesso!")
                                
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Modelo", result["model_type"].title())
                                col2.metric("RMSE", f"{result['rmse']:.4f}")
                                col3.metric("MAE", f"{result['mae']:.4f}")
                                
                                # Plot forecast
                                if result.get("forecast"):
                                    import plotly.graph_objects as go
                                    
                                    fig = go.Figure()
                                    
                                    # Historical data
                                    fig.add_trace(go.Scatter(
                                        x=result["historical_dates"],
                                        y=result["historical_values"],
                                        mode="lines",
                                        name="Dados Históricos",
                                        line=dict(color="blue")
                                    ))
                                    
                                    # Forecast
                                    fig.add_trace(go.Scatter(
                                        x=result["forecast_dates"],
                                        y=result["forecast"],
                                        mode="lines",
                                        name="Previsão",
                                        line=dict(color="red")
                                    ))
                                    
                                    # Confidence intervals (se disponível)
                                    if result.get("forecast_lower") and result.get("forecast_upper"):
                                        fig.add_trace(go.Scatter(
                                            x=result["forecast_dates"],
                                            y=result["forecast_upper"],
                                            mode="lines",
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=result["forecast_dates"],
                                            y=result["forecast_lower"],
                                            mode="lines",
                                            line=dict(width=0),
                                            fill="tonexty",
                                            fillcolor="rgba(255,0,0,0.2)",
                                            name="Intervalo de Confiança"
                                        ))
                                    
                                    fig.update_layout(
                                        title=f"Previsão - {result['model_type'].title()}",
                                        xaxis_title="Data",
                                        yaxis_title="Valor",
                                        hovermode="x unified"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"timeseries_forecast_{model_type}_{model_name}")
                                
                                # Salvar modelo
                                model_name = f"{model_type}_ts_model"
                                if st.button("Salvar Modelo", key=f"save_ts_{model_name}"):
                                    try:
                                        saved_path = ts_automl.save_model(result, model_name)
                                        st.success(f"Modelo salvo em: {saved_path}")
                                    except Exception as exc:
                                        st.error(f"Erro ao salvar modelo: {exc}")
                                
                            except Exception as exc:
                                st.error(f"Erro no treinamento: {exc}")
                
                except Exception as exc:
                    st.error(f"Erro ao ler dataset: {exc}")
        
        with ts_tabs[1]:
            st.subheader("Modelos Time Series Salvos")
            
            models = ts_automl.list_models()
            
            if models:
                models_df = pd.DataFrame(models)
                st.dataframe(models_df, use_container_width=True)
                
                # Detalhes do modelo selecionado
                selected_model = st.selectbox(
                    "Ver detalhes do modelo",
                    options=[m["name"] for m in models],
                    key="ts_model_select"
                )
                
                if selected_model:
                    model_info = next((m for m in models if m["name"] == selected_model), None)
                    if model_info:
                        st.write("**Detalhes do Modelo:**")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Modelo", model_info["model_type"].title())
                        col2.metric("RMSE", f"{model_info['rmse']:.4f}")
                        col3.metric("MAE", f"{model_info['mae']:.4f}")
                        
                        # Botões de ação
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Carregar Modelo", key=f"load_ts_{selected_model}"):
                                try:
                                    loaded = ts_automl.load_model(model_info["path"])
                                    st.success("Modelo carregado com sucesso!")
                                except Exception as exc:
                                    st.error(f"Erro ao carregar modelo: {exc}")
                        
                        with col2:
                            if st.button("Excluir Modelo", key=f"delete_ts_{selected_model}"):
                                try:
                                    import shutil
                                    shutil.rmtree(model_info["path"])
                                    st.success("Modelo excluído com sucesso!")
                                    st.experimental_rerun()
                                except Exception as exc:
                                    st.error(f"Erro ao excluir modelo: {exc}")
            else:
                st.info("Nenhum modelo Time Series salvo ainda.")
        
        with ts_tabs[2]:
            st.subheader("Previsões com Modelos Time Series")
            
            models = ts_automl.list_models()
            
            if models:
                # Selecionar modelo
                model_options = {m["name"]: m for m in models}
                selected_pred_model = st.selectbox(
                    "Modelo para previsão",
                    options=list(model_options.keys()),
                    key="ts_pred_model_select"
                )
                
                if selected_pred_model:
                    model_info = model_options[selected_pred_model]
                    
                    # Upload de dados para previsão
                    pred_file = st.file_uploader("Upload Dataset para Previsão (CSV)", type=["csv"])
                    
                    if pred_file is not None:
                        try:
                            pred_df = pd.read_csv(pred_file)
                            st.write(f"Dataset de previsão: {pred_df.shape}")
                            st.dataframe(pred_df.head(), use_container_width=True)
                            
                            # Configurações de previsão
                            periods = st.number_input("Período de Previsão (dias)", min_value=1, max_value=365, value=30)
                            frequency = st.selectbox("Frequência", options=["D", "W", "M"], index=0)
                            
                            # Upload de dados históricos (opcional para LSTM)
                            if model_info["model_type"] == "lstm":
                                st.write("**Dados Históricos (para LSTM):**")
                                hist_file = st.file_uploader("Upload Dataset Histórico (CSV)", type=["csv"])
                                
                                if hist_file is not None:
                                    try:
                                        hist_df = pd.read_csv(hist_file)
                                        st.write(f"Dataset histórico: {hist_df.shape}")
                                        st.dataframe(hist_df.head(), use_container_width=True)
                                        
                                        # Selecionar colunas
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            hist_date_col = st.selectbox("Coluna de Data Histórica", options=list(hist_df.columns))
                                        with col2:
                                            hist_value_col = st.selectbox("Coluna de Valor Histórica", options=[c for c in hist_df.columns if c != hist_date_col])
                                    except Exception as exc:
                                        st.error(f"Erro ao ler dataset histórico: {exc}")
                        except Exception as exc:
                            st.error(f"Erro ao ler dataset de previsão: {exc}")
                    
                    if st.button("Realizar Previsões", type="primary"):
                        with st.spinner("Realizando previsões..."):
                            try:
                                if model_info["model_type"] == "lstm" and 'hist_df' in locals():
                                    # Preparar dados históricos
                                    hist_ts_df = hist_df[[hist_date_col, hist_value_col]].copy()
                                    hist_ts_df[hist_date_col] = pd.to_datetime(hist_ts_df[hist_date_col])
                                    hist_ts_df = hist_ts_df.sort_values(hist_date_col)
                                    hist_ts_df = hist_ts_df.set_index(hist_date_col)
                                    hist_series = hist_ts_df[hist_value_col].asfreq('D').dropna()
                                    
                                    result = ts_automl.forecast(
                                        model_info["path"], periods, frequency, hist_series
                                    )
                                else:
                                    # ARIMA e Prophet - não precisam de dados históricos
                                    result = ts_automl.forecast(
                                        model_info["path"], periods, frequency
                                    )
                                
                                if result["success"]:
                                    st.success("✅ Previsões realizadas com sucesso!")
                                    
                                    # Criar DataFrame de resultados
                                    if model_info["model_type"] == "prophet":
                                        results_df = pd.DataFrame({
                                            "data": result["dates"],
                                            "previsao": result["forecast"],
                                            "limite_inferior": result["forecast_lower"],
                                            "limite_superior": result["forecast_upper"],
                                        })
                                    else:
                                        # ARIMA
                                        last_date = pd.Timestamp.now()
                                        forecast_dates = pd.date_range(
                                            start=last_date + pd.Timedelta(days=1),
                                            periods=periods,
                                            freq=frequency
                                        )
                                        
                                        results_df = pd.DataFrame({
                                            "data": forecast_dates,
                                            "previsao": result["forecast"],
                                        })
                                        
                                        if result["forecast_ci"] is not None:
                                            results_df["limite_inferior"] = [ci[0] for ci in result["forecast_ci"]]
                                            results_df["limite_superior"] = [ci[1] for ci in result["forecast_ci"]]
                                    
                                    st.write("**Resultados da Previsão:**")
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Download
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="Baixar Previsões",
                                        data=csv,
                                        file_name=f"previsoes_{selected_pred_model}.csv",
                                        mime="text/csv",
                                    )
                                else:
                                    st.error(f"Erro na previsão: {result['error']}")
                                
                            except Exception as exc:
                                st.error(f"Erro na previsão: {exc}")
                    else:
                        st.info("👆 Por favor, faça upload dos dados e selecione a coluna target primeiro.")
            else:
                st.info("Nenhum modelo Time Series disponível. Treine um modelo primeiro.")
        
        with ts_tabs[3]:
            st.subheader("Análise de Séries Temporais")
            
            # Upload de dados para análise
            analysis_file = st.file_uploader("Upload Dataset para Análise (CSV)", type=["csv"])
            
            if analysis_file is not None:
                try:
                    df = pd.read_csv(analysis_file)
                    st.write(f"Dataset carregado: {df.shape}")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Selecionar colunas
                    col1, col2 = st.columns(2)
                    with col1:
                        date_column = st.selectbox("Coluna de Data", options=list(df.columns))
                    with col2:
                        value_column = st.selectbox("Coluna de Valor", options=[c for c in df.columns if c != date_column])
                    
                    if st.button("Analisar Série Temporal", type="primary"):
                        with st.spinner("Realizando análise..."):
                            try:
                                # Preparar dados
                                ts_df = df[[date_column, value_column]].copy()
                                ts_df[date_column] = pd.to_datetime(ts_df[date_column])
                                ts_df = ts_df.sort_values(date_column)
                                ts_df = ts_df.set_index(date_column)
                                ts_series = ts_df[value_column].asfreq('D').dropna()
                                
                                if len(ts_series) < 30:
                                    st.error("Dados insuficientes para análise (mínimo 30 pontos).")
                                    return
                                
                                # Estatísticas básicas
                                st.write("**Estatísticas Básicas:**")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Observações", len(ts_series))
                                col2.metric("Média", f"{ts_series.mean():.4f}")
                                col3.metric("Desvio Padrão", f"{ts_series.std():.4f}")
                                col4.metric("Range", f"{ts_series.max() - ts_series.min():.4f}")
                                
                                # Gráfico da série
                                import plotly.graph_objects as go
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=ts_series.index,
                                    y=ts_series.values,
                                    mode="lines",
                                    name="Série Temporal",
                                    line=dict(color="blue")
                                ))
                                
                                fig.update_layout(
                                    title="Série Temporal",
                                    xaxis_title="Data",
                                    yaxis_title="Valor",
                                    hovermode="x unified"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"timeseries_plot_{model_type}_{model_name}")
                                
                            except Exception as exc:
                                st.error(f"Erro na análise: {exc}")
                
                except Exception as exc:
                    st.error(f"Erro ao ler dataset: {exc}")

    with tabs[9]:
        st.subheader("Monitoramento")
        
        # Selecionar modelo para monitorar
        try:
            experiments = list_experiment_records(settings=settings, limit=200, offset=0)
            if experiments:
                model_options = {f"{e['id']} – {e['best_model_name']}": e for e in experiments}
                selected_label = st.selectbox("Selecione um modelo para monitorar", options=list(model_options.keys()), key="monitor_model_select")
                selected_exp = model_options[selected_label]
                
                # Inicializar monitor
                monitor = get_performance_monitor(f"model_{selected_exp['id']}")
                
                st.write("### Métricas de Performance")
                
                # Log de predição manual para teste
                with st.expander("Testar Log de Predição"):
                    col1, col2 = st.columns(2)
                    with col1:
                        prediction = st.number_input("Predição", value=1.0)
                        ground_truth = st.number_input("Ground Truth", value=1.0)
                    with col2:
                        latency_ms = st.number_input("Latência (ms)", value=100.0)
                    
                    if st.button("Registrar Predição"):
                        monitor.log_prediction({"test": "data"}, prediction, ground_truth, latency_ms)
                        st.success("Predição registrada!")
                
                # Mostrar métricas
                metrics = monitor.get_metrics()
                if metrics:
                    st.json(metrics)
                else:
                    st.info("Nenhuma métrica registrada ainda")
                
                # Alertas
                st.write("### Configurar Alertas")
                alert_manager = get_alert_manager()
                
                with st.expander("Configurar Thresholds"):
                    accuracy_threshold = st.slider("Threshold de Accuracy", 0.0, 1.0, 0.8)
                    latency_threshold = st.slider("Threshold de Latência (ms)", 0, 1000, 500)
                    
                    if st.button("Salvar Configurações"):
                        st.success("Configurações salvas!")
                
            else:
                st.info("Nenhum experimento encontrado para monitorar")
                
        except Exception as exc:
            st.error(f"Erro no monitoramento: {exc}")

    with tabs[10]:
        st.subheader("Deploy local (API)")
        st.write("Para subir a API localmente:")
        st.code("python -m free_mlops.api")
        st.write("Endpoints:")
        st.code(f"http://{settings.api_host}:{settings.api_port}/docs")


if __name__ == "__main__":
    main()
