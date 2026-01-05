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


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def main() -> None:
    st.set_page_config(page_title="Free MLOps", layout="wide")

    settings = get_settings()

    st.title("Free MLOps (MVP)")

    tabs = st.tabs(["Treinar", "Experimentos", "Model Registry", "Fine-Tune", "Deploy/API"])

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

    with tabs[4]:
        st.subheader("Deploy local (API)")
        st.write("Para subir a API localmente:")
        st.code("python -m free_mlops.api")
        st.write("Endpoints:")
        st.code(f"http://{settings.api_host}:{settings.api_port}/docs")


if __name__ == "__main__":
    main()
