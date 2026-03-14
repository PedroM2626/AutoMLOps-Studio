from __future__ import annotations

import reflex as rx

from .state import AppState

MODULES = [
    "Overview",
    "Data",
    "AutoML",
    "Experiments",
    "Registry & Deploy",
    "Monitoring",
    "Computer Vision",
]

DARK_BG = "#090c12"
DARK_SURFACE = "#121826"
DARK_CARD = "#171f2f"
DARK_BORDER = "#283247"
TEXT_PRIMARY = "#e7eefb"
TEXT_MUTED = "#9dafcf"
ACCENT = "#4f8cff"
ACCENT_2 = "#32c9a8"
ACCENT_3 = "#ffb84d"


def card(*children, **kwargs) -> rx.Component:
    style = {
        "background": "linear-gradient(180deg, rgba(23,31,47,0.98) 0%, rgba(15,22,36,0.98) 100%)",
        "border": f"1px solid {DARK_BORDER}",
        "border_radius": "18px",
        "padding": "1.1rem",
        "box_shadow": "0 18px 40px rgba(0,0,0,0.28)",
        "backdrop_filter": "blur(18px)",
        "transition": "transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease",
        "_hover": {
            "transform": "translateY(-2px)",
            "box_shadow": "0 24px 54px rgba(0,0,0,0.34)",
            "border_color": "rgba(79, 140, 255, 0.45)",
        },
    }
    style.update(kwargs)
    return rx.box(*children, **style)


def label(text: str) -> rx.Component:
    return rx.text(text, color=TEXT_MUTED, font_size="0.82rem", margin_bottom="0.25rem")


def input_block(title: str, component: rx.Component) -> rx.Component:
    return rx.vstack(label(title), component, spacing="1", align_items="start", width="100%")


def status_chip(text_var: rx.Var, kind_var: rx.Var) -> rx.Component:
    return rx.box(
        text_var,
        padding="0.2rem 0.65rem",
        border_radius="999px",
        font_size="0.78rem",
        font_weight="700",
        color=rx.cond(kind_var == "online", "#0b2a21", "#261b0b"),
        background=rx.cond(kind_var == "online", "#72f1cf", "#ffd89b"),
    )


def section_title(title: str, subtitle: str) -> rx.Component:
    return rx.vstack(
        rx.text(title, color=TEXT_PRIMARY, font_size="1.02rem", font_weight="800"),
        rx.text(subtitle, color=TEXT_MUTED, font_size="0.82rem"),
        spacing="1",
        align_items="start",
        width="100%",
    )


def manual_param_cards() -> rx.Component:
    return rx.vstack(
        section_title("Manual Parameters by Algorithm", "Cada algoritmo agora usa campos visuais interativos, sincronizados automaticamente com o payload final."),
        rx.foreach(
            AppState.automl_manual_param_cards,
            lambda block: card(
                rx.hstack(
                    rx.vstack(
                        rx.text(block["model_label"], color=TEXT_PRIMARY, font_weight="800"),
                        rx.text(block["model"], color=TEXT_MUTED, font_size="0.74rem"),
                        spacing="0",
                        align_items="start",
                    ),
                    rx.spacer(),
                    rx.box(
                        rx.text(block["field_count"], color="#041a17", font_weight="800", font_size="0.76rem"),
                        padding="0.2rem 0.55rem",
                        border_radius="999px",
                        background=ACCENT_2,
                    ),
                    width="100%",
                    align_items="center",
                ),
                rx.grid(
                    rx.foreach(
                        block["fields"],
                        lambda field: input_block(
                            field["label"],
                            rx.vstack(
                                rx.cond(
                                    field["kind"] == "select",
                                    rx.select(
                                        field["options"],
                                        value=field["value_text"],
                                        on_change=lambda value: AppState.update_manual_param_value(block["model"], field["name"], value),
                                        width="100%",
                                    ),
                                    rx.cond(
                                        field["kind"] == "int",
                                        rx.input(
                                            type="number",
                                            value=field["value_text"],
                                            on_change=lambda value: AppState.update_manual_param_value(block["model"], field["name"], value),
                                            width="100%",
                                        ),
                                        rx.cond(
                                            field["kind"] == "float",
                                            rx.input(
                                                type="number",
                                                value=field["value_text"],
                                                on_change=lambda value: AppState.update_manual_param_value(block["model"], field["name"], value),
                                                width="100%",
                                            ),
                                            rx.input(
                                                value=field["value_text"],
                                                on_change=lambda value: AppState.update_manual_param_value(block["model"], field["name"], value),
                                                width="100%",
                                            ),
                                        ),
                                    ),
                                ),
                                rx.cond(
                                    field["kind"] == "select",
                                    rx.text("Options: ", field["options_text"], color=TEXT_MUTED, font_size="0.72rem"),
                                    rx.cond(
                                        (field["min_value"] != "") | (field["max_value"] != ""),
                                        rx.text("Range: ", field["min_value"], " to ", field["max_value"], color=TEXT_MUTED, font_size="0.72rem"),
                                    ),
                                ),
                                spacing="1",
                                align_items="start",
                                width="100%",
                            ),
                        ),
                    ),
                    columns="2",
                    spacing="3",
                    width="100%",
                    margin_top="0.8rem",
                ),
            ),
        ),
        width="100%",
        spacing="3",
        align_items="start",
    )


def sidebar() -> rx.Component:
    def nav_btn(name: str) -> rx.Component:
        return rx.button(
            name,
            width="100%",
            justify_content="start",
            padding="0.85rem 0.95rem",
            background=rx.cond(AppState.active_module == name, "linear-gradient(90deg, rgba(79,140,255,0.24), rgba(50,201,168,0.12))", "transparent"),
            border=rx.cond(AppState.active_module == name, "1px solid rgba(79,140,255,0.55)", "1px solid rgba(255,255,255,0.04)"),
            color=TEXT_PRIMARY,
            border_radius="12px",
            transition="all 180ms ease",
            _hover={"background": "rgba(79,140,255,0.12)", "transform": "translateX(3px)"},
            on_click=lambda: AppState.set_module(name),
        )

    return rx.box(
        rx.vstack(
            rx.box(
                rx.text("AutoMLOps Studio", font_size="1.35rem", font_weight="900", color=TEXT_PRIMARY),
                rx.text("Reflex Native Console", font_size="0.82rem", color=TEXT_MUTED),
                padding="0.95rem",
                border_radius="16px",
                width="100%",
                background="linear-gradient(145deg, rgba(79,140,255,0.22), rgba(50,201,168,0.08))",
                border=f"1px solid {DARK_BORDER}",
            ),
            rx.divider(border_color=DARK_BORDER),
            rx.vstack(*[nav_btn(name) for name in MODULES], width="100%", spacing="2"),
            rx.spacer(),
            width="100%",
            align_items="start",
            min_height="100%",
        ),
        width=["100%", "100%", "280px"],
        height="100vh",
        background=DARK_SURFACE,
        border_right=f"1px solid {DARK_BORDER}",
        padding="1rem",
        overflow_y="auto",
        overflow_x="hidden",
        flex_shrink="0",
    )





def overview_page() -> rx.Component:
    return rx.vstack(
        rx.grid(
            card(
                rx.text("Datasets", color=TEXT_MUTED),
                rx.heading(AppState.dataset_count, size="6", color=TEXT_PRIMARY),
                rx.text("Collections in Data Lake", color=TEXT_MUTED, font_size="0.8rem"),
            ),
            card(
                rx.text("Versions", color=TEXT_MUTED),
                rx.heading(AppState.dataset_version_count, size="6", color=TEXT_PRIMARY),
                rx.text("Versioned files", color=TEXT_MUTED, font_size="0.8rem"),
            ),
            card(
                rx.text("MLflow Runs", color=TEXT_MUTED),
                rx.heading(AppState.run_count, size="6", color=TEXT_PRIMARY),
                rx.text("Tracked experiments", color=TEXT_MUTED, font_size="0.8rem"),
            ),
            card(
                rx.text("Registered Models", color=TEXT_MUTED),
                rx.heading(AppState.registered_model_count, size="6", color=TEXT_PRIMARY),
                rx.text("Registry entries", color=TEXT_MUTED, font_size="0.8rem"),
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        rx.grid(
            card(
                rx.hstack(
                    rx.vstack(
                        rx.text("Serving API", color=TEXT_PRIMARY, font_weight="700"),
                        rx.text(AppState.api_url, color=TEXT_MUTED, font_size="0.8rem"),
                        align_items="start",
                        spacing="1",
                    ),
                    rx.spacer(),
                    status_chip(AppState.api_status_label, AppState.api_status),
                    width="100%",
                ),
            ),
            card(
                rx.hstack(
                    rx.vstack(
                        rx.text("MLflow", color=TEXT_PRIMARY, font_weight="700"),
                        rx.text(AppState.mlflow_url, color=TEXT_MUTED, font_size="0.8rem"),
                        align_items="start",
                        spacing="1",
                    ),
                    rx.spacer(),
                    status_chip(AppState.mlflow_status_label, AppState.mlflow_status),
                    width="100%",
                ),
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        card(
            rx.text("Recent Runs", color=TEXT_PRIMARY, font_weight="700", margin_bottom="0.6rem"),
            rx.foreach(
                AppState.runs,
                lambda item: rx.box(
                    rx.hstack(
                        rx.text(item["experiment"], color=TEXT_PRIMARY, font_weight="700"),
                        rx.spacer(),
                        rx.text(item["status"], color=TEXT_MUTED),
                        width="100%",
                    ),
                    rx.text(item["metrics"], color=TEXT_MUTED, font_size="0.78rem"),
                    padding="0.55rem",
                    border_bottom=f"1px solid {DARK_BORDER}",
                ),
            ),
        ),
        card(
            rx.text("DagsHub Integration", color=TEXT_PRIMARY, font_weight="700"),
            rx.text("Conecte o tracking do MLflow ao repositorio remoto no DagsHub.", color=TEXT_MUTED, font_size="0.82rem"),
            rx.grid(
                input_block("DagsHub Username", rx.input(value=AppState.dagshub_username, on_change=AppState.set_dagshub_username, width="100%")),
                input_block("Repository Name", rx.input(value=AppState.dagshub_repo, on_change=AppState.set_dagshub_repo, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            input_block("DagsHub Token", rx.input(value=AppState.dagshub_token, on_change=AppState.set_dagshub_token, type="password", width="100%")),
            rx.hstack(
                rx.button("Connect", on_click=AppState.connect_dagshub_tracking, background=ACCENT_2, color="#041a17"),
                rx.button("Disconnect", on_click=AppState.disconnect_dagshub_tracking, background="#3b4357", color=TEXT_PRIMARY),
                spacing="2",
            ),
            rx.text(AppState.dagshub_status_label, color=rx.cond(AppState.dagshub_connected, ACCENT_2, TEXT_MUTED), font_weight="700"),
            rx.text(AppState.tracking_uri, color=TEXT_MUTED, font_size="0.78rem"),
            rx.cond(AppState.dagshub_message != "", rx.text(AppState.dagshub_message, color=TEXT_PRIMARY, font_size="0.82rem")),
        ),
        spacing="4",
        width="100%",
    )


def data_page() -> rx.Component:
    return rx.vstack(
        card(
            rx.text("Data Ingestion", color=TEXT_PRIMARY, font_weight="700"),
            rx.text("Anexe um ou mais arquivos (CSV, JSON, Parquet, TXT, ZIP) para salvar no Data Lake.", color=TEXT_MUTED, font_size="0.82rem"),
            rx.upload(
                rx.vstack(
                    rx.button("Select File(s)", background=ACCENT_2, color="#041a17"),
                    rx.text("Drag and drop files here", color=TEXT_MUTED, font_size="0.8rem"),
                    spacing="2",
                    align_items="center",
                    width="100%",
                ),
                id="data_upload",
                multiple=True,
                width="100%",
                border=f"1px dashed {DARK_BORDER}",
                padding="1rem",
                border_radius="10px",
            ),
            rx.hstack(
                rx.button(
                    "Upload to Data Lake",
                    on_click=AppState.handle_data_upload(rx.upload_files(upload_id="data_upload")),
                    background=ACCENT,
                    color="#06122d",
                ),
                rx.button("Refresh", on_click=AppState.refresh, background="#3b4357", color=TEXT_PRIMARY),
                spacing="2",
            ),
            rx.foreach(
                rx.selected_files("data_upload"),
                lambda file: rx.text(file, color=TEXT_MUTED, font_size="0.78rem"),
            ),
            rx.cond(AppState.data_upload_status != "", rx.text(AppState.data_upload_status, color=ACCENT_2, font_size="0.82rem")),
        ),
        card(
            rx.text("Path Upload (optional)", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Local file path", rx.input(value=AppState.local_file_path, on_change=AppState.set_local_file_path, placeholder="C:/path/file.csv", width="100%")),
                input_block("Dataset name", rx.input(value=AppState.dataset_name, on_change=AppState.set_dataset_name, placeholder="dataset_slug", width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.button("Save file to Data Lake", on_click=AppState.save_local_file_to_lake, background=ACCENT, color="#06122d"),
        ),
        card(
            rx.text("Schema & Split Controls", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block(
                    "Split strategy",
                    rx.select(
                        ["Random", "Chronological", "Manual (Pre-defined split column)"],
                        value=AppState.data_split_strategy,
                        on_change=AppState.set_data_split_strategy,
                        width="100%",
                    ),
                ),
                input_block("Train percentage", rx.input(type="number", value=AppState.data_train_percent, on_change=AppState.set_data_train_percent, width="100%")),
                input_block("Time column (chrono)", rx.select(AppState.data_column_options_optional, placeholder="— none —", value=AppState.data_time_column, on_change=AppState.set_data_time_column, width="100%")),
                input_block("Manual split column", rx.select(AppState.data_column_options_optional, placeholder="— none —", value=AppState.data_manual_split_column, on_change=AppState.set_data_manual_split_column, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            input_block(
                "Schema overrides JSON",
                rx.text_area(
                    value=AppState.data_schema_overrides_json,
                    on_change=AppState.set_data_schema_overrides_json,
                    min_height="120px",
                    width="100%",
                ),
            ),
            rx.text("Use o mesmo formato da interface Streamlit: lista de objetos com Include/Column/Type.", color=TEXT_MUTED, font_size="0.78rem"),
        ),
        card(
            rx.text("Dataset Preview", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Dataset", rx.select(AppState.dataset_options, value=AppState.selected_dataset, on_change=AppState.update_selected_dataset, width="100%")),
                input_block("Version", rx.select(AppState.selected_dataset_version_options, value=AppState.selected_version, on_change=AppState.update_selected_version, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.hstack(
                rx.button("Load Preview", on_click=AppState.preview_dataset, background=ACCENT_2, color="#041a17"),
                rx.button("Delete Version", on_click=AppState.delete_selected_version, background="#763a43", color=TEXT_PRIMARY),
            ),
            rx.code_block(AppState.dataset_preview_md, language="markdown", width="100%"),
        ),
        card(
            rx.text("Drift Detection", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Reference dataset", rx.select(AppState.dataset_options, value=AppState.drift_reference_dataset, on_change=AppState.update_drift_reference_dataset, width="100%")),
                input_block("Reference version", rx.select(AppState.drift_reference_version_options, value=AppState.drift_reference_version, on_change=AppState.set_drift_reference_version, width="100%")),
                input_block("Current dataset", rx.select(AppState.dataset_options, value=AppState.drift_current_dataset, on_change=AppState.update_drift_current_dataset, width="100%")),
                input_block("Current version", rx.select(AppState.drift_current_version_options, value=AppState.drift_current_version, on_change=AppState.set_drift_current_version, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.button("Run Drift", on_click=AppState.run_drift_detection, background=ACCENT, color="#06122d"),
            rx.code_block(AppState.drift_results_json, language="json", width="100%"),
        ),
        spacing="4",
        width="100%",
    )


def automl_page() -> rx.Component:
    return rx.vstack(
        card(
            section_title("Classical AutoML Training", "Selecione modelos-base, ajuste validação e monte ensembles customizados sem misturar estratégias com algoritmos."),
            rx.grid(
                input_block(
                    "Task",
                    rx.select(
                        ["classification", "regression", "clustering", "time_series", "anomaly_detection"],
                        value=AppState.automl_task,
                        on_change=AppState.update_automl_task,
                        width="100%",
                    ),
                ),
                input_block("Target column", rx.select(AppState.automl_target_options, value=AppState.automl_target_column, on_change=AppState.set_automl_target_column, width="100%")),
                input_block("Experiment name", rx.input(value=AppState.automl_experiment_name, on_change=AppState.set_automl_experiment_name, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.grid(
                input_block("Train dataset", rx.select(AppState.dataset_options, value=AppState.automl_train_dataset, on_change=AppState.update_automl_train_dataset, width="100%")),
                input_block("Train version", rx.select(AppState.automl_train_version_options, value=AppState.automl_train_version, on_change=AppState.update_automl_train_version, width="100%")),
                input_block("Test dataset (optional)", rx.select(AppState.dataset_options_optional, placeholder="— none —", value=AppState.automl_test_dataset, on_change=AppState.update_automl_test_dataset, width="100%")),
                input_block("Test version (optional)", rx.select(AppState.automl_test_version_options_optional, placeholder="— none —", value=AppState.automl_test_version, on_change=AppState.set_automl_test_version, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.grid(
                input_block(
                    "Model Source",
                    rx.select(
                        ["Standard AutoML", "Model Registry", "Local Upload"],
                        value=AppState.automl_model_source,
                        on_change=AppState.update_model_source,
                        width="100%",
                    ),
                ),
                input_block(
                    "Model Selection",
                    rx.select(
                        ["Automatic (Preset)", "Manual (Select)"],
                        value=AppState.automl_mode_selection,
                        on_change=AppState.update_mode_selection,
                        width="100%",
                    ),
                ),
                input_block(
                    "Training Strategy",
                    rx.select(
                        ["Automatic", "Manual"],
                        value=AppState.automl_training_strategy,
                        on_change=AppState.update_training_strategy,
                        width="100%",
                    ),
                ),
                input_block(
                    "Training Focus",
                    rx.select(
                        ["single", "ensemble_only", "both"],
                        value=AppState.automl_ensemble_mode,
                        on_change=AppState.update_ensemble_mode,
                        width="100%",
                    ),
                ),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.cond(
                AppState.automl_model_source == "Model Registry",
                input_block("Registry model name", rx.select(AppState.model_name_options, value=AppState.automl_registry_model_name, on_change=AppState.set_automl_registry_model_name, width="100%")),
            ),
            rx.cond(
                AppState.automl_model_source == "Local Upload",
                rx.vstack(
                    rx.upload(
                        rx.vstack(
                            rx.button("Select model file (.pkl/.joblib/.onnx)", background=ACCENT_2, color="#041a17"),
                            rx.text("Drop model file here", color=TEXT_MUTED, font_size="0.8rem"),
                            spacing="2",
                            align_items="center",
                            width="100%",
                        ),
                        id="model_upload",
                        multiple=False,
                        width="100%",
                        border=f"1px dashed {DARK_BORDER}",
                        padding="1rem",
                        border_radius="10px",
                    ),
                    rx.button(
                        "Upload model",
                        on_click=AppState.handle_model_upload(rx.upload_files(upload_id="model_upload")),
                        background=ACCENT,
                        color="#06122d",
                    ),
                    rx.cond(AppState.automl_uploaded_model_status != "", rx.text(AppState.automl_uploaded_model_status, color=ACCENT_2, font_size="0.82rem")),
                    spacing="2",
                    width="100%",
                    align_items="start",
                ),
            ),
            rx.checkbox("Include Deep Learning Models", is_checked=AppState.automl_use_deep_learning, on_change=AppState.update_use_deep_learning),
            rx.grid(
                input_block("Preset", rx.select(["test", "fast", "medium", "high", "custom"], value=AppState.automl_preset, on_change=AppState.set_automl_preset, width="100%")),
                input_block("Optimization mode", rx.select(["bayesian", "random", "grid", "hyperband"], value=AppState.automl_optimization_mode, on_change=AppState.set_automl_optimization_mode, width="100%")),
                input_block(
                    "Optimization metric",
                    rx.select(
                        AppState.automl_metric_options,
                        value=AppState.automl_optimization_metric,
                        on_change=AppState.set_automl_optimization_metric,
                        width="100%",
                    ),
                ),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.grid(
                input_block("Trials", rx.input(type="number", value=AppState.automl_n_trials, on_change=AppState.set_automl_n_trials_input, width="100%")),
                input_block("Timeout (s)", rx.input(type="number", value=AppState.automl_timeout, on_change=AppState.set_automl_timeout_input, width="100%")),
                input_block("Time budget (s)", rx.input(type="number", value=AppState.automl_time_budget, on_change=AppState.set_automl_time_budget_input, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            card(
                section_title("Advanced Validation", "A Reflex agora usa os nomes corretos do trainer e expõe parâmetros extras que eram do wizard Streamlit."),
                rx.grid(
                    input_block(
                        "Validation strategy",
                        rx.select(
                            ["auto", "cv", "stratified_cv", "holdout", "auto_split", "time_series_cv"],
                            value=AppState.automl_validation_strategy,
                            on_change=AppState.update_validation_strategy,
                            width="100%",
                        ),
                    ),
                    input_block("Validation folds", rx.input(type="number", value=AppState.automl_validation_folds, on_change=AppState.set_automl_validation_folds_input, width="100%")),
                    input_block("Holdout test size (%)", rx.input(type="number", value=AppState.automl_validation_test_size, on_change=AppState.set_automl_validation_test_size_input, width="100%")),
                    input_block("Time-series gap", rx.input(type="number", value=AppState.automl_validation_gap, on_change=AppState.set_automl_validation_gap_input, width="100%")),
                    input_block("Max train window (0 = full)", rx.input(type="number", value=AppState.automl_validation_max_train_size, on_change=AppState.set_automl_validation_max_train_size_input, width="100%")),
                    input_block("Random seed", rx.input(type="number", value=AppState.automl_random_state, on_change=AppState.set_automl_random_state_input, width="100%")),
                    input_block("Early stopping", rx.input(type="number", value=AppState.automl_early_stopping, on_change=AppState.set_automl_early_stopping_input, width="100%")),
                    columns="2",
                    spacing="3",
                    width="100%",
                    margin_top="0.85rem",
                ),
                rx.hstack(
                    rx.checkbox("Shuffle CV / holdout", is_checked=AppState.automl_validation_shuffle, on_change=AppState.set_automl_validation_shuffle),
                    rx.checkbox("Stratify holdout", is_checked=AppState.automl_validation_stratify, on_change=AppState.set_automl_validation_stratify),
                    spacing="4",
                    margin_top="0.55rem",
                ),
                rx.text("Use holdout/auto_split para test size; use cv/stratified_cv para folds; use time_series_cv para gap e janela máxima.", color=TEXT_MUTED, font_size="0.78rem", margin_top="0.45rem"),
            ),
            rx.cond(
                AppState.automl_mode_selection == "Manual (Select)",
                card(
                    section_title("Base Model Picker", "Voting, stacking e bagging saem daqui e ficam apenas no ensemble builder."),
                    rx.flex(
                        rx.foreach(
                            AppState.automl_available_models,
                            lambda model_name: rx.button(
                                model_name,
                                on_click=lambda: AppState.toggle_automl_model(model_name),
                                border_radius="999px",
                                padding="0.35rem 0.75rem",
                                font_size="0.78rem",
                                background=rx.cond(model_name == "", "transparent", rx.cond(AppState.automl_selected_model_list.contains(model_name), "linear-gradient(90deg, rgba(79,140,255,0.95), rgba(50,201,168,0.8))", "#1b2539")),
                                color=rx.cond(AppState.automl_selected_model_list.contains(model_name), "#051325", TEXT_PRIMARY),
                                border=rx.cond(AppState.automl_selected_model_list.contains(model_name), "1px solid rgba(50,201,168,0.6)", f"1px solid {DARK_BORDER}"),
                                transition="all 160ms ease",
                                _hover={"transform": "translateY(-1px)", "border_color": "rgba(79,140,255,0.55)"},
                            ),
                        ),
                        wrap="wrap",
                        spacing="2",
                        margin_top="0.85rem",
                    ),
                    input_block(
                        "Selected models (comma separated)",
                        rx.text_area(
                            value=AppState.automl_models_csv,
                            on_change=AppState.set_automl_models_csv_input,
                            min_height="90px",
                            width="100%",
                        ),
                    ),
                    rx.text("Clique nos chips para incluir/remover modelos ou cole uma lista separada por vírgulas.", color=TEXT_MUTED, font_size="0.78rem"),
                ),
            ),
            rx.cond(
                AppState.automl_training_strategy == "Manual",
                rx.vstack(
                    rx.cond(AppState.automl_manual_param_cards != [], manual_param_cards()),
                    card(
                        rx.text("Manual Params Payload (auto-generated)", color=TEXT_PRIMARY, font_weight="700"),
                        rx.code_block(AppState.automl_manual_params_json, language="json", width="100%"),
                    ),
                    width="100%",
                    spacing="3",
                    align_items="start",
                ),
            ),
            card(
                section_title("Custom Ensemble Builder", "Configure strategies explicitamente aqui, sem poluir a seleção de modelos-base."),
                rx.grid(
                    input_block(
                        "Ensemble type",
                        rx.select(["Voting", "Stacking", "Bagging"], value=AppState.new_ensemble_type, on_change=AppState.set_new_ensemble_type, width="100%"),
                    ),
                    input_block("Base models (comma separated)", rx.input(value=AppState.new_ensemble_models_csv, on_change=AppState.set_new_ensemble_models_csv, width="100%")),
                    input_block("Meta model (stacking)", rx.select(AppState.automl_available_models, value=AppState.new_ensemble_meta_model, on_change=AppState.set_new_ensemble_meta_model, width="100%")),
                    input_block("Voting type", rx.select(["soft", "hard"], value=AppState.new_ensemble_voting_type, on_change=AppState.set_new_ensemble_voting_type, width="100%")),
                    input_block("Bagging base", rx.select(AppState.automl_available_models, value=AppState.new_ensemble_bagging_base, on_change=AppState.set_new_ensemble_bagging_base, width="100%")),
                    columns="2",
                    spacing="3",
                    width="100%",
                ),
                rx.button("Add ensemble", on_click=AppState.add_custom_ensemble, background=ACCENT_2, color="#041a17"),
                rx.foreach(
                    AppState.automl_custom_ensembles,
                    lambda item: card(
                        rx.hstack(
                            rx.text(
                                item["type"],
                                " | models: ",
                                item["models"],
                                " | meta: ",
                                item["meta_model"],
                                " | voting: ",
                                item["voting_type"],
                                color=TEXT_MUTED,
                                font_size="0.78rem",
                            ),
                            rx.spacer(),
                            rx.button("Remove", on_click=lambda: AppState.remove_custom_ensemble(item["id"]), background="#522b32", color=TEXT_PRIMARY, size="1"),
                            width="100%",
                        ),
                        padding="0.8rem",
                    ),
                ),
            ),
            rx.hstack(
                rx.button("Refresh available models", on_click=AppState.refresh_models_for_task, background="#3b4357", color=TEXT_PRIMARY),
                rx.button("Submit AutoML Job", on_click=AppState.submit_automl, background=ACCENT, color="#06122d"),
                spacing="2",
            ),
            rx.text("Eligible base models:", color=TEXT_MUTED),
            rx.flex(
                rx.foreach(
                    AppState.automl_available_models,
                    lambda model_name: rx.box(
                        model_name,
                        border=f"1px solid {DARK_BORDER}",
                        border_radius="999px",
                        padding="0.25rem 0.6rem",
                        color=TEXT_PRIMARY,
                        font_size="0.75rem",
                        background="#1f2a42",
                    ),
                ),
                wrap="wrap",
                spacing="2",
            ),
            rx.text(AppState.automl_submit_status, color=ACCENT_2),
        ),
        spacing="4",
        width="100%",
    )


def experiments_page() -> rx.Component:
    return rx.vstack(
        card(
            rx.text("Training Results Chart", color=TEXT_PRIMARY, font_weight="700"),
            rx.text("Top jobs por melhor score (normalizado).", color=TEXT_MUTED, font_size="0.82rem"),
            rx.foreach(
                AppState.job_chart_rows,
                lambda row: rx.vstack(
                    rx.hstack(
                        rx.text(row["name"], color=TEXT_PRIMARY, font_size="0.8rem"),
                        rx.spacer(),
                        rx.text(row["score"], color=ACCENT_2, font_size="0.8rem"),
                        width="100%",
                    ),
                    rx.box(
                        rx.box(height="8px", width=row["width"], background=ACCENT, border_radius="999px"),
                        height="8px",
                        width="100%",
                        background="#1f2a42",
                        border_radius="999px",
                    ),
                    spacing="1",
                    width="100%",
                    align_items="start",
                ),
            ),
        ),
        card(
            rx.hstack(
                rx.text("Background Training Jobs", color=TEXT_PRIMARY, font_weight="700"),
                rx.spacer(),
                rx.button("Refresh Jobs", on_click=AppState.refresh_jobs, background=ACCENT, color="#06122d"),
                width="100%",
            ),
            rx.foreach(
                AppState.jobs,
                lambda item: rx.box(
                    rx.hstack(
                        rx.text(item["job_id"], color=TEXT_PRIMARY, font_weight="700"),
                        rx.text(item["name"], color=TEXT_MUTED),
                        rx.spacer(),
                        rx.text(item["status_label"], color=TEXT_PRIMARY),
                        rx.text(item["best_score"], color=ACCENT_2),
                        width="100%",
                    ),
                    rx.text("Duration: ", item["duration"], color=TEXT_MUTED, font_size="0.78rem"),
                    rx.text("Run: ", item["mlflow_run_id"], color=TEXT_MUTED, font_size="0.74rem"),
                    border_bottom=f"1px solid {DARK_BORDER}",
                    padding="0.55rem",
                ),
            ),
            input_block("Selected job id", rx.select(AppState.job_id_options, value=AppState.selected_job_id, on_change=AppState.update_selected_job, width="100%")),
            rx.hstack(
                rx.button("Pause", on_click=AppState.pause_selected_job, background="#6b5b30", color=TEXT_PRIMARY),
                rx.button("Resume", on_click=AppState.resume_selected_job, background="#2f5f53", color=TEXT_PRIMARY),
                rx.button("Cancel", on_click=AppState.cancel_selected_job, background="#763a43", color=TEXT_PRIMARY),
                rx.button("Delete", on_click=AppState.delete_selected_job, background="#522b32", color=TEXT_PRIMARY),
                spacing="2",
            ),
            rx.text("Logs (tail)", color=TEXT_MUTED),
            rx.code_block(AppState.selected_job_logs, language="log", width="100%"),
            rx.cond(AppState.selected_job_error != "", rx.code_block(AppState.selected_job_error, language="log", width="100%")),
        ),
        spacing="4",
        width="100%",
    )


def registry_deploy_page() -> rx.Component:
    return rx.vstack(
        card(
            rx.text("Model Registry", color=TEXT_PRIMARY, font_weight="700"),
            rx.foreach(
                AppState.registry_rows,
                lambda item: rx.box(
                    rx.hstack(
                        rx.text(item["name"], color=TEXT_PRIMARY, font_weight="700"),
                        rx.spacer(),
                        rx.text(item["stage"], color=TEXT_MUTED),
                        rx.text("v", item["version"], color=ACCENT_2),
                        width="100%",
                    ),
                    rx.text(item["description"], color=TEXT_MUTED, font_size="0.8rem"),
                    border_bottom=f"1px solid {DARK_BORDER}",
                    padding="0.5rem",
                ),
            ),
            rx.grid(
                input_block("Model name", rx.select(AppState.model_name_options, value=AppState.selected_model_name, on_change=AppState.update_selected_model_name, width="100%")),
                input_block("Model version (optional)", rx.select(AppState.selected_model_version_options_optional, placeholder="— none —", value=AppState.selected_model_version, on_change=AppState.set_selected_model_version, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.button("Load model details", on_click=AppState.load_model_details, background=ACCENT, color="#06122d"),
            rx.code_block(AppState.model_details_json, language="json", width="100%"),
        ),
        card(
            rx.text("Register Model from Run", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Run ID", rx.select(AppState.run_id_options, value=AppState.selected_run_id, on_change=AppState.set_selected_run_id, width="100%")),
                input_block("New model name", rx.input(value=AppState.register_model_name, on_change=AppState.set_register_model_name, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.hstack(
                rx.button("Load run details", on_click=AppState.load_run_details, background="#3b4357", color=TEXT_PRIMARY),
                rx.button("Register run", on_click=AppState.register_selected_run, background=ACCENT_2, color="#041a17"),
                spacing="2",
            ),
            rx.text(AppState.register_result, color=ACCENT_2),
            rx.code_block(AppState.run_details_json, language="json", width="100%"),
        ),
        card(
            rx.text("Deploy to Hugging Face", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Model path", rx.input(value=AppState.hf_model_path, on_change=AppState.set_hf_model_path, width="100%")),
                input_block("Repo id (user/repo)", rx.input(value=AppState.hf_repo_id, on_change=AppState.set_hf_repo_id, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            input_block("HF token", rx.input(value=AppState.hf_token, on_change=AppState.set_hf_token, type="password", width="100%")),
            rx.checkbox("Private repo", is_checked=AppState.hf_private, on_change=AppState.set_hf_private),
            rx.button("Deploy", on_click=AppState.deploy_to_hf, background=ACCENT, color="#06122d"),
            rx.code_block(AppState.hf_result, language="log", width="100%"),
        ),
        card(
            rx.text("API Playground", color=TEXT_PRIMARY, font_weight="700"),
            input_block("Predict URL", rx.input(value=AppState.api_predict_url, on_change=AppState.set_api_predict_url, width="100%")),
            input_block("API Key", rx.input(value=AppState.api_key, on_change=AppState.set_api_key, type="password", width="100%")),
            input_block("Payload JSON", rx.text_area(value=AppState.api_payload_json, on_change=AppState.set_api_payload_json, min_height="120px", width="100%")),
            rx.button("Send Request", on_click=AppState.send_predict_request, background=ACCENT_2, color="#041a17"),
            rx.code_block(AppState.api_response_json, language="json", width="100%"),
        ),
        spacing="4",
        width="100%",
    )


def monitoring_page() -> rx.Component:
    return rx.vstack(
        card(
            rx.text("Telemetry", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                card(rx.text("Rows", color=TEXT_MUTED), rx.heading(AppState.telemetry_row_count, size="5", color=TEXT_PRIMARY), padding="0.7rem"),
                card(rx.text("Last seen", color=TEXT_MUTED), rx.text(AppState.telemetry_last_seen, color=TEXT_PRIMARY), padding="0.7rem"),
                columns="2",
                spacing="3",
                width="100%",
            ),
        ),
        card(
            rx.text("Model Stability", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Model name", rx.select(AppState.model_name_options, value=AppState.stability_model_name, on_change=AppState.update_stability_model_name, width="100%")),
                input_block("Version (optional)", rx.select(AppState.stability_model_version_options_optional, placeholder="— none —", value=AppState.stability_model_version, on_change=AppState.set_stability_model_version, width="100%")),
                input_block("Dataset", rx.select(AppState.dataset_options, value=AppState.stability_dataset, on_change=AppState.update_stability_dataset, width="100%")),
                input_block("Dataset version", rx.select(AppState.stability_version_options, value=AppState.stability_version, on_change=AppState.update_stability_version, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.grid(
                input_block("Target column", rx.select(AppState.stability_target_options, value=AppState.stability_target, on_change=AppState.set_stability_target, width="100%")),
                input_block("Task", rx.select(["classification", "regression"], value=AppState.stability_task, on_change=AppState.set_stability_task, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.hstack(
                rx.checkbox("Seed stability", is_checked=AppState.stability_seed, on_change=AppState.set_stability_seed),
                rx.checkbox("Split stability", is_checked=AppState.stability_split, on_change=AppState.set_stability_split),
                rx.checkbox("Noise robustness", is_checked=AppState.stability_noise, on_change=AppState.set_stability_noise),
                spacing="4",
            ),
            rx.button("Run Stability", on_click=AppState.run_stability, background=ACCENT, color="#06122d"),
            rx.code_block(AppState.stability_result_json, language="json", width="100%"),
        ),
        spacing="4",
        width="100%",
    )


def cv_page() -> rx.Component:
    return rx.vstack(
        card(
            rx.text("Computer Vision Training", color=TEXT_PRIMARY, font_weight="700"),
            rx.grid(
                input_block("Task", rx.select(["image_classification", "image_multi_label", "image_segmentation", "object_detection"], value=AppState.cv_task_type, on_change=AppState.set_cv_task_type, width="100%")),
                input_block("Backbone", rx.select(["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121", "vgg16"], value=AppState.cv_backbone, on_change=AppState.set_cv_backbone, width="100%")),
                input_block("Num classes", rx.input(type="number", value=AppState.cv_num_classes, on_change=AppState.set_cv_num_classes_input, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.grid(
                input_block("Data directory", rx.select(AppState.cv_data_dir_options_optional, placeholder="— select —", value=AppState.cv_data_dir, on_change=AppState.update_cv_data_dir, width="100%")),
                input_block("Label CSV (optional)", rx.select(AppState.cv_label_csv_options_optional, placeholder="— none —", value=AppState.cv_label_csv, on_change=AppState.set_cv_label_csv, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.grid(
                input_block("Epochs", rx.input(type="number", value=AppState.cv_epochs, on_change=AppState.set_cv_epochs_input, width="100%")),
                input_block("Batch size", rx.input(type="number", value=AppState.cv_batch_size, on_change=AppState.set_cv_batch_size_input, width="100%")),
                input_block("Learning rate", rx.input(type="number", value=AppState.cv_learning_rate, on_change=AppState.set_cv_learning_rate_input, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.button("Run CV Training", on_click=AppState.run_cv_training, background=ACCENT_2, color="#041a17"),
            rx.code_block(AppState.cv_result_json, language="json", width="100%"),
        ),
        spacing="4",
        width="100%",
    )


def module_view() -> rx.Component:
    return rx.cond(
        AppState.active_module == "Overview",
        overview_page(),
        rx.cond(
            AppState.active_module == "Data",
            data_page(),
            rx.cond(
                AppState.active_module == "AutoML",
                automl_page(),
                rx.cond(
                    AppState.active_module == "Experiments",
                    experiments_page(),
                    rx.cond(
                        AppState.active_module == "Registry & Deploy",
                        registry_deploy_page(),
                        rx.cond(
                            AppState.active_module == "Monitoring",
                            monitoring_page(),
                            cv_page(),
                        ),
                    ),
                ),
            ),
        ),
    )


def shell() -> rx.Component:
    return rx.box(
        rx.box(
            position="fixed",
            top="-120px",
            right="-40px",
            width="320px",
            height="320px",
            border_radius="999px",
            background="radial-gradient(circle, rgba(79,140,255,0.22) 0%, rgba(79,140,255,0) 72%)",
            pointer_events="none",
        ),
        rx.box(
            position="fixed",
            bottom="-140px",
            left="20%",
            width="360px",
            height="360px",
            border_radius="999px",
            background="radial-gradient(circle, rgba(50,201,168,0.16) 0%, rgba(50,201,168,0) 72%)",
            pointer_events="none",
        ),
        sidebar(),
        rx.box(
            rx.box(module_view(), width="100%"),
            padding="1.2rem",
            width="100%",
            height="100vh",
            background=DARK_BG,
            overflow_y="auto",
            overflow_x="hidden",
        ),
        display="flex",
        height="100vh",
        background=DARK_BG,
        overflow="hidden",
    )

