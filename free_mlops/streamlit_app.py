from __future__ import annotations

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


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def main() -> None:
    st.set_page_config(page_title="Free MLOps", layout="wide")

    settings = get_settings()

    st.title("Free MLOps (MVP)")

    tabs = st.tabs(["Treinar", "Experimentos", "Model Registry", "Testar Modelos", "Monitoramento", "Hyperopt", "DVC", "Data Validation", "Fine-Tune", "Deploy/API"])

    with tabs[0]:
        st.subheader("1) Upload do CSV")

        uploaded = st.file_uploader("Dataset (CSV)", type=["csv"])

        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            file_hash = hash_uploaded_file(file_bytes)

            if st.session_state.get("uploaded_hash") != file_hash:
                dataset_path = save_uploaded_bytes(
                    file_bytes=file_bytes,
                    original_filename=uploaded.name,
                    settings=settings,
                )
                st.session_state["uploaded_hash"] = file_hash
                st.session_state["dataset_path"] = str(dataset_path)
                st.session_state.pop("last_experiment", None)

        dataset_path_str = st.session_state.get("dataset_path")
        if dataset_path_str:
            dataset_path = Path(dataset_path_str)
            try:
                df = load_csv(dataset_path)
            except Exception as exc:
                st.error(f"Falha ao ler CSV: {exc}")
                st.stop()

            st.subheader("2) Selecionar target e tipo")
            st.write("Prévia do dataset")
            st.dataframe(df.head(200), use_container_width=True)

            target_column = st.selectbox("Target (coluna alvo)", options=list(df.columns))
            problem_type = st.radio(
                "Tipo do problema",
                options=["classification", "regression", "multiclass_classification", "binary_classification"],
                horizontal=True,
            )

            # Gerar ID do experimento antes de treinar
            if "experiment_id" not in st.session_state:
                st.session_state["experiment_id"] = uuid4().hex

            st.info(f"ID do experimento: {st.session_state['experiment_id']}")

            # Personalização do treinamento
            with st.expander("Personalizar treinamento"):
                all_models = [
                    "logistic_regression",
                    "linear_svc",
                    "random_forest",
                    "extra_trees",
                    "gradient_boosting",
                    "decision_tree",
                    "knn",
                    "ridge",
                    "lasso",
                    "elastic_net",
                    "svr",
                ]
                selected_models = st.multiselect(
                    "Algoritmos para treinar",
                    options=all_models,
                    default=all_models,
                )
                max_time = st.number_input(
                    "Tempo máximo de treinamento (segundos, 0 = ilimitado)",
                    min_value=0,
                    value=0,
                    step=30,
                )
                max_time_seconds = None if max_time == 0 else max_time

            st.subheader("3) Treinar (AutoML)")
            train_clicked = st.button("Treinar", type="primary")

            if train_clicked:
                with st.spinner("Treinando e avaliando modelos..."):
                    try:
                        record = run_experiment(
                            dataset_path=dataset_path,
                            target_column=target_column,
                            problem_type=problem_type,
                            settings=settings,
                            selected_models=selected_models,
                            max_time_seconds=max_time_seconds,
                        )
                        st.session_state["last_experiment"] = record
                        st.session_state.pop("experiment_id", None)  # Limpar ID para novo experimento
                    except Exception as exc:
                        st.error(str(exc))

            record = st.session_state.get("last_experiment")
            if record:
                st.subheader("Resultados")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Experiment ID", record["id"])
                col2.metric("Melhor modelo", record["best_model_name"])
                col3.metric("Tipo", record["problem_type"])
                col4.metric("Tempo de treino", f"{record.get('model_metadata', {}).get('training_time_seconds', 0):.2f}s")

                st.write("Métricas detalhadas")
                st.json(record["best_metrics"])

                st.write("Leaderboard")
                st.dataframe(pd.DataFrame(record.get("leaderboard", [])), use_container_width=True)

                model_path = Path(record["model_path"])
                if model_path.exists():
                    st.download_button(
                        label="Baixar modelo (.pkl)",
                        data=_read_bytes(model_path),
                        file_name=f"model_{record['id']}.pkl",
                        mime="application/octet-stream",
                        key=f"download_{record['id']}",
                    )

                # Botão para registrar modelo
                if st.button("Registrar modelo no Model Registry", key=f"register_{record['id']}"):
                    try:
                        new_record = register_model(
                            settings=settings,
                            experiment_id=record["id"],
                            new_version="v1.1.0",
                            description="Modelo registrado via UI",
                        )
                        st.success(f"Modelo registrado: {new_record['id']}")
                    except Exception as exc:
                        st.error(f"Erro ao registrar modelo: {exc}")

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

            selected_id = st.selectbox("Ver detalhes", options=[e["id"] for e in experiments])
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
            selected_label = st.selectbox("Escolha um modelo para testar", options=list(model_options.keys()))
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
            selected_label = st.selectbox("Escolha um experimento para fine-tune", options=list(exp_options.keys()))
            selected_exp = exp_options[selected_label]

            st.write("Detalhes do experimento base")
            st.json(selected_exp["best_metrics"])

            model_name = st.selectbox(
                "Modelo para fine-tune",
                options=[
                    "logistic_regression",
                    "linear_svc",
                    "random_forest",
                    "extra_trees",
                    "gradient_boosting",
                    "decision_tree",
                    "knn",
                    "ridge",
                    "lasso",
                    "elastic_net",
                    "svr",
                ],
                index=0,
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
                target_column = st.selectbox("Target (coluna alvo)", options=list(df.columns))
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
                
                selected_model = st.selectbox("Modelo para otimizar", options=available_models)
                
                # Configurações da otimização
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_trials = st.number_input("Número de trials", min_value=10, max_value=1000, value=100)
                with col2:
                    cv_folds = st.number_input("CV Folds", min_value=2, max_value=10, value=5)
                with col3:
                    timeout = st.number_input("Timeout (segundos, 0 = ilimitado)", min_value=0, value=300, step=60)
                
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
                        dataset_type = st.selectbox("Tipo", options=["raw", "processed", "models"])
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
                            filter_type = st.selectbox("Filtrar por tipo", options=["todos", "raw", "processed", "models"])
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
                        export_format = st.selectbox("Formato", options=["json", "python"])
                        
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
                    schema1 = st.selectbox("Schema 1", options=[s["name"] for s in schemas])
                with col2:
                    schema2 = st.selectbox("Schema 2", options=[s["name"] for s in schemas])
                
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
        st.subheader("Deploy local (API)")
        st.write("Para subir a API localmente:")
        st.code("python -m free_mlops.api")
        st.write("Endpoints:")
        st.code(f"http://{settings.api_host}:{settings.api_port}/docs")


if __name__ == "__main__":
    main()
