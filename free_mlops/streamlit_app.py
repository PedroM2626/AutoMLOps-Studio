from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from free_mlops.config import get_settings
from free_mlops.service import hash_uploaded_file
from free_mlops.service import list_experiment_records
from free_mlops.service import load_csv
from free_mlops.service import run_experiment
from free_mlops.service import save_uploaded_bytes


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def main() -> None:
    st.set_page_config(page_title="Free MLOps", layout="wide")

    settings = get_settings()

    st.title("Free MLOps (MVP)")

    tabs = st.tabs(["Treinar", "Experimentos", "Deploy/API"])

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
                options=["classification", "regression"],
                horizontal=True,
            )

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
                        )
                        st.session_state["last_experiment"] = record
                    except Exception as exc:
                        st.error(str(exc))

            record = st.session_state.get("last_experiment")
            if record:
                st.subheader("Resultados")

                col1, col2, col3 = st.columns(3)
                col1.metric("Experiment ID", record["id"])
                col2.metric("Melhor modelo", record["best_model_name"])

                best_metrics = record.get("best_metrics", {})
                if problem_type == "classification":
                    col3.metric("F1 (weighted)", f"{best_metrics.get('f1_weighted', 'n/a')}")
                else:
                    col3.metric("RMSE", f"{best_metrics.get('rmse', 'n/a')}")

                st.write("Métricas do melhor modelo")
                st.json(best_metrics)

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

    with tabs[2]:
        st.subheader("Deploy local (API)")
        st.write("Para subir a API localmente:")
        st.code("python -m free_mlops.api")
        st.write("Endpoints:")
        st.code(f"http://{settings.api_host}:{settings.api_port}/docs")


if __name__ == "__main__":
    main()
