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
DARK_SURFACE = "#0f1420"
DARK_CARD = "#141c2e"
DARK_BORDER = "#1e2d47"
TEXT_PRIMARY = "#e7eefb"
TEXT_MUTED = "#7a90b8"
ACCENT = "#4f8cff"
ACCENT_2 = "#32c9a8"
ACCENT_3 = "#ffb84d"
ACCENT_RED = "#f04f5f"

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #090c12; }
::-webkit-scrollbar-thumb { background: #1e2d47; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4f8cff44; }
.log-line-info  { color: #58a6ff; }
.log-line-warn  { color: #ffb84d; }
.log-line-error { color: #f04f5f; font-weight: 700; }
.log-line-ok    { color: #32c9a8; }
.log-line-def   { color: #7a90b8; }
"""


def card(*children, **kwargs) -> rx.Component:
    style = {
        "background": "linear-gradient(160deg, rgba(20,28,46,0.98) 0%, rgba(12,18,32,0.98) 100%)",
        "border": f"1px solid {DARK_BORDER}",
        "border_radius": "16px",
        "padding": "1.1rem",
        "box_shadow": "0 12px 32px rgba(0,0,0,0.32)",
        "backdrop_filter": "blur(18px)",
        "transition": "transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease",
        "_hover": {
            "transform": "translateY(-2px)",
            "box_shadow": "0 20px 48px rgba(0,0,0,0.40)",
            "border_color": "rgba(79,140,255,0.5)",
        },
    }
    style.update(kwargs)
    return rx.box(*children, **style)


def label(text: str) -> rx.Component:
    return rx.text(text, color=TEXT_MUTED, font_size="0.80rem", margin_bottom="0.2rem", font_weight="500")


def input_block(title: str, component: rx.Component) -> rx.Component:
    return rx.vstack(label(title), component, spacing="1", align_items="start", width="100%")


def status_chip(text_var: rx.Var, kind_var: rx.Var) -> rx.Component:
    return rx.box(
        text_var,
        padding="0.2rem 0.7rem",
        border_radius="999px",
        font_size="0.75rem",
        font_weight="700",
        color=rx.cond(kind_var == "online", "#041a17", rx.cond(kind_var == "starting", "#1a1400", "#1a0a0f")),
        background=rx.cond(kind_var == "online", ACCENT_2, rx.cond(kind_var == "starting", ACCENT_3, "#f04f5f")),
    )


def section_title(title: str, subtitle: str) -> rx.Component:
    return rx.vstack(
        rx.text(title, color=TEXT_PRIMARY, font_size="1.05rem", font_weight="800"),
        rx.text(subtitle, color=TEXT_MUTED, font_size="0.80rem"),
        spacing="1",
        align_items="start",
        width="100%",
        margin_bottom="0.5rem",
    )


def hero_header(title: str, subtitle: str) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(
                title,
                font_size="2rem",
                font_weight="900",
                background="linear-gradient(90deg, #4f8cff 0%, #32c9a8 60%, #a78bfa 100%)",
                background_clip="text",
                color="transparent",
                style={"-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"},
                line_height="1.15",
            ),
            rx.text(subtitle, color=TEXT_MUTED, font_size="0.88rem", margin_top="2px"),
            spacing="1",
            align_items="start",
        ),
        padding="0.6rem 0 1rem 0",
        width="100%",
    )


def stat_card(value_var: rx.Var, label_text: str, accent: str = ACCENT) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(value_var, size="7", color=TEXT_PRIMARY, font_weight="900"),
            rx.text(label_text, color=TEXT_MUTED, font_size="0.72rem", font_weight="600",
                    text_transform="uppercase", letter_spacing="0.8px"),
            spacing="1",
            align_items="start",
        ),
        background=f"linear-gradient(160deg, rgba(20,28,46,0.98) 0%, rgba(12,18,32,0.98) 100%)",
        border=f"1px solid {DARK_BORDER}",
        border_left=f"3px solid {accent}",
        border_radius="12px",
        padding="0.9rem 1rem",
        transition="transform 200ms ease, box-shadow 200ms ease",
        _hover={"transform": "translateY(-2px)", "box_shadow": f"0 8px 24px {accent}22"},
    )


def badge(text_var, kind: str = "info") -> rx.Component:
    colors = {
        "running": ("#27ae60", "rgba(39,174,96,0.15)", "rgba(39,174,96,0.3)"),
        "done": (ACCENT, "rgba(79,140,255,0.15)", "rgba(79,140,255,0.3)"),
        "failed": (ACCENT_RED, "rgba(240,79,95,0.15)", "rgba(240,79,95,0.3)"),
        "queued": ("#a78bfa", "rgba(167,139,250,0.15)", "rgba(167,139,250,0.3)"),
        "paused": (ACCENT_3, "rgba(255,184,77,0.15)", "rgba(255,184,77,0.3)"),
        "info": (ACCENT, "rgba(79,140,255,0.15)", "rgba(79,140,255,0.3)"),
    }
    fg, bg, border = colors.get(kind, colors["info"])
    return rx.box(
        text_var,
        display="inline-block",
        padding="0.15rem 0.6rem",
        border_radius="999px",
        font_size="0.72rem",
        font_weight="700",
        letter_spacing="0.3px",
        color=fg,
        background=bg,
        border=f"1px solid {border}",
    )


def colored_log_view(log_text_var: rx.Var) -> rx.Component:
    """Styled terminal box for logs - shows log text with monospace font."""
    return rx.box(
        rx.text(
            log_text_var,
            white_space="pre-wrap",
            word_break="break-all",
            font_family="'Courier New', Consolas, monospace",
            font_size="0.76rem",
            line_height="1.7",
            color="#8b949e",
        ),
        background="#050810",
        border=f"1px solid {DARK_BORDER}",
        border_radius="10px",
        padding="1rem",
        max_height="400px",
        overflow_y="auto",
        width="100%",
    )


def service_row(label_text: str, url_var: rx.Var, status_var: rx.Var, status_label_var: rx.Var, service_key: str) -> rx.Component:
    return rx.hstack(
        rx.vstack(
            rx.text(label_text, color=TEXT_PRIMARY, font_weight="700", font_size="0.88rem"),
            rx.text(url_var, color=TEXT_MUTED, font_size="0.74rem"),
            spacing="0",
            align_items="start",
        ),
        rx.spacer(),
        status_chip(status_label_var, status_var),
        rx.button(
            "Start",
            on_click=lambda: AppState.start_service(service_key),
            background="rgba(79,140,255,0.15)",
            color=ACCENT,
            border=f"1px solid rgba(79,140,255,0.3)",
            border_radius="8px",
            padding="0.3rem 0.8rem",
            font_size="0.78rem",
            font_weight="600",
            _hover={"background": "rgba(79,140,255,0.25)"},
            size="1",
        ),
        rx.button(
            "Stop",
            on_click=lambda: AppState.stop_service(service_key),
            background="rgba(240,79,95,0.12)",
            color=ACCENT_RED,
            border=f"1px solid rgba(240,79,95,0.25)",
            border_radius="8px",
            padding="0.3rem 0.8rem",
            font_size="0.78rem",
            font_weight="600",
            _hover={"background": "rgba(240,79,95,0.22)"},
            size="1",
        ),
        width="100%",
        align_items="center",
        padding="0.6rem 0",
        border_bottom=f"1px solid {DARK_BORDER}",
    )


def manual_param_cards() -> rx.Component:
    return rx.vstack(
        section_title("Manual Parameters by Algorithm", "Each algorithm now uses interactive visual fields, automatically synchronized with the final payload."),
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
        icons = {
            "Overview": "home", "Data": "database", "AutoML": "brain",
            "Experiments": "flask-conical", "Registry & Deploy": "package",
            "Monitoring": "activity", "Computer Vision": "eye",
        }
        return rx.button(
            rx.hstack(
                rx.icon(icons.get(name, "circle"), size=14, color=rx.cond(AppState.active_module == name, ACCENT, TEXT_MUTED)),
                rx.text(name, font_size="0.84rem", font_weight=rx.cond(AppState.active_module == name, "700", "500")),
                spacing="2",
                align_items="center",
            ),
            width="100%",
            justify_content="start",
            padding="0.7rem 0.95rem",
            background=rx.cond(AppState.active_module == name, "linear-gradient(90deg, rgba(79,140,255,0.18), rgba(50,201,168,0.08))", "transparent"),
            border=rx.cond(AppState.active_module == name, f"1px solid rgba(79,140,255,0.4)", "1px solid transparent"),
            color=rx.cond(AppState.active_module == name, TEXT_PRIMARY, TEXT_MUTED),
            border_radius="10px",
            transition="all 180ms ease",
            _hover={"background": "rgba(79,140,255,0.10)", "color": TEXT_PRIMARY, "transform": "translateX(3px)"},
            on_click=lambda: AppState.set_module(name),
        )

    return rx.box(
        rx.vstack(
            # Brand header
            rx.box(
                rx.hstack(
                    rx.box(
                        rx.icon("bot", size=20, color=ACCENT),
                        background="rgba(79,140,255,0.15)",
                        border_radius="8px",
                        padding="0.4rem",
                    ),
                    rx.vstack(
                        rx.text("AutoMLOps Studio", font_size="0.95rem", font_weight="900", color=TEXT_PRIMARY, line_height="1"),
                        rx.hstack(
                            rx.text("v4.7.1", font_size="0.65rem", color=ACCENT, font_weight="700"),
                            rx.text("Open Source", font_size="0.65rem", color=TEXT_MUTED),
                            spacing="2",
                        ),
                        spacing="0",
                        align_items="start",
                    ),
                    spacing="2",
                    align_items="center",
                ),
                padding="0.8rem",
                border_radius="12px",
                width="100%",
                background="linear-gradient(145deg, rgba(79,140,255,0.1), rgba(50,201,168,0.05))",
                border=f"1px solid {DARK_BORDER}",
            ),
            rx.divider(border_color=DARK_BORDER, margin_y="0.3rem"),
            # Navigation
            rx.vstack(*[nav_btn(name) for name in MODULES], width="100%", spacing="1"),
            rx.divider(border_color=DARK_BORDER, margin_y="0.3rem"),
            # System Overview stats
            rx.text("SYSTEM OVERVIEW", font_size="0.62rem", color=TEXT_MUTED, font_weight="700", letter_spacing="1.2px"),
            rx.grid(
                rx.box(
                    rx.heading(AppState.dataset_count, size="5", color=TEXT_PRIMARY, font_weight="800"),
                    rx.text("Datasets", color=TEXT_MUTED, font_size="0.68rem"),
                    border_left=f"2px solid {ACCENT}",
                    padding_left="0.5rem",
                ),
                rx.box(
                    rx.heading(AppState.run_count, size="5", color=TEXT_PRIMARY, font_weight="800"),
                    rx.text("Runs", color=TEXT_MUTED, font_size="0.68rem"),
                    border_left=f"2px solid {ACCENT_2}",
                    padding_left="0.5rem",
                ),
                rx.box(
                    rx.heading(AppState.registered_model_count, size="5", color=TEXT_PRIMARY, font_weight="800"),
                    rx.text("Models", color=TEXT_MUTED, font_size="0.68rem"),
                    border_left=f"2px solid #a78bfa",
                    padding_left="0.5rem",
                ),
                rx.box(
                    rx.heading(AppState.dataset_version_count, size="5", color=TEXT_PRIMARY, font_weight="800"),
                    rx.text("Versions", color=TEXT_MUTED, font_size="0.68rem"),
                    border_left=f"2px solid {ACCENT_3}",
                    padding_left="0.5rem",
                ),
                columns="2",
                spacing="3",
                width="100%",
            ),
            rx.divider(border_color=DARK_BORDER, margin_y="0.3rem"),
            # Service statuses
            rx.text("SERVICES", font_size="0.62rem", color=TEXT_MUTED, font_weight="700", letter_spacing="1.2px"),
            rx.vstack(
                rx.hstack(
                    rx.text("Serving API", color=TEXT_MUTED, font_size="0.76rem"),
                    rx.spacer(),
                    status_chip(AppState.api_status_label, AppState.api_status),
                    width="100%",
                    align_items="center",
                ),
                rx.hstack(
                    rx.text("MLflow", color=TEXT_MUTED, font_size="0.76rem"),
                    rx.spacer(),
                    status_chip(AppState.mlflow_status_label, AppState.mlflow_status),
                    width="100%",
                    align_items="center",
                ),
                spacing="2",
                width="100%",
            ),
            rx.divider(border_color=DARK_BORDER, margin_y="0.3rem"),
            # DagsHub integration
            rx.text("DAGSHUB", font_size="0.62rem", color=TEXT_MUTED, font_weight="700", letter_spacing="1.2px"),
            rx.vstack(
                input_block("Username", rx.input(value=AppState.dagshub_username, on_change=AppState.set_dagshub_username, size="1", width="100%", placeholder="dagshub_user")),
                input_block("Repository", rx.input(value=AppState.dagshub_repo, on_change=AppState.set_dagshub_repo, size="1", width="100%", placeholder="repo_name")),
                input_block("Token", rx.input(value=AppState.dagshub_token, on_change=AppState.set_dagshub_token, type="password", size="1", width="100%", placeholder="api_token")),
                rx.hstack(
                    rx.button("Connect", on_click=AppState.connect_dagshub_tracking, background=ACCENT_2, color="#041a17", size="1", font_weight="700"),
                    rx.button("Disconnect", on_click=AppState.disconnect_dagshub_tracking, background="rgba(240,79,95,0.12)", color=ACCENT_RED, border=f"1px solid rgba(240,79,95,0.25)", size="1"),
                    spacing="2",
                    width="100%",
                ),
                rx.text(
                    AppState.dagshub_status_label,
                    color=rx.cond(AppState.dagshub_connected, ACCENT_2, TEXT_MUTED),
                    font_size="0.74rem", font_weight="700",
                ),
                rx.cond(
                    AppState.dagshub_message != "",
                    rx.text(AppState.dagshub_message, color=TEXT_MUTED, font_size="0.70rem"),
                ),
                spacing="2",
                width="100%",
                align_items="start",
            ),
            rx.text(AppState.tracking_uri, color=TEXT_MUTED, font_size="0.62rem", word_break="break-all"),
            rx.button(
                rx.hstack(rx.icon("refresh-cw", size=12), rx.text("Refresh All", font_size="0.76rem"), spacing="1", align_items="center"),
                on_click=AppState.refresh,
                width="100%",
                background="rgba(79,140,255,0.08)",
                color=TEXT_MUTED,
                border=f"1px solid {DARK_BORDER}",
                border_radius="8px",
                _hover={"background": "rgba(79,140,255,0.16)", "color": TEXT_PRIMARY},
                margin_top="0.5rem",
            ),
            width="100%",
            align_items="start",
            spacing="2",
        ),
        width=["100%", "100%", "290px"],
        height="100vh",
        background=DARK_SURFACE,
        border_right=f"1px solid {DARK_BORDER}",
        padding="0.9rem",
        overflow_y="auto",
        overflow_x="hidden",
        flex_shrink="0",
    )





def overview_page() -> rx.Component:
    return rx.vstack(
        hero_header("AutoMLOps Studio", "Automated Machine Learning & MLOps Platform"),
        # Stat cards
        rx.grid(
            stat_card(AppState.dataset_count, "Datasets", ACCENT),
            stat_card(AppState.dataset_version_count, "Versions", ACCENT_2),
            stat_card(AppState.run_count, "MLflow Runs", "#a78bfa"),
            stat_card(AppState.registered_model_count, "Registered Models", ACCENT_3),
            columns="4",
            spacing="4",
            width="100%",
        ),
        # Services
        card(
            section_title("Infrastructure Services", "Start and monitor background services for serving and experiment tracking."),
            service_row("Serving API", AppState.api_url, AppState.api_status, AppState.api_status_label, "api"),
            service_row("MLflow Tracking UI", AppState.mlflow_url, AppState.mlflow_status, AppState.mlflow_status_label, "mlflow"),
            rx.hstack(
                rx.link(
                    rx.button(
                        rx.hstack(rx.icon("external-link", size=12), rx.text("Open MLflow UI", font_size="0.78rem"), spacing="1"),
                        background="rgba(79,140,255,0.1)", color=ACCENT, border=f"1px solid rgba(79,140,255,0.25)",
                        border_radius="8px", size="1",
                        _hover={"background": "rgba(79,140,255,0.2)"},
                    ),
                    href=AppState.mlflow_url,
                    is_external=True,
                ),
                rx.link(
                    rx.button(
                        rx.hstack(rx.icon("external-link", size=12), rx.text("Open API Docs", font_size="0.78rem"), spacing="1"),
                        background="rgba(50,201,168,0.1)", color=ACCENT_2, border=f"1px solid rgba(50,201,168,0.25)",
                        border_radius="8px", size="1",
                        _hover={"background": "rgba(50,201,168,0.2)"},
                    ),
                    href=AppState.api_url,
                    is_external=True,
                ),
                rx.button(
                    rx.hstack(rx.icon("refresh-cw", size=12), rx.text("Refresh", font_size="0.78rem"), spacing="1"),
                    on_click=AppState.refresh,
                    background="rgba(255,255,255,0.04)", color=TEXT_MUTED,
                    border=f"1px solid {DARK_BORDER}", border_radius="8px", size="1",
                    _hover={"color": TEXT_PRIMARY},
                ),
                spacing="2",
                margin_top="0.6rem",
            ),
        ),
        # Datasets catalog
        card(
            section_title("Dataset Catalog", "Datasets stored in the Data Lake."),
            rx.cond(
                AppState.datasets.length() > 0,
                rx.vstack(
                    rx.hstack(
                        rx.text("NAME", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px", min_width="140px"),
                        rx.text("VERSIONS", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px", min_width="80px"),
                        rx.text("LATEST", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px", min_width="120px"),
                        rx.text("COLUMNS (PREVIEW)", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px"),
                        width="100%",
                        padding="0.4rem 0.5rem",
                        border_bottom=f"1px solid {DARK_BORDER}",
                    ),
                    rx.foreach(
                        AppState.datasets,
                        lambda row: rx.hstack(
                            rx.hstack(
                                rx.icon("database", size=13, color=ACCENT),
                                rx.text(row["name"], color=TEXT_PRIMARY, font_weight="600", font_size="0.84rem"),
                                spacing="1",
                                min_width="140px",
                            ),
                            rx.text(row["versions"], color=ACCENT_2, font_weight="700", font_size="0.82rem", min_width="80px"),
                            rx.text(row["latest_updated"], color=TEXT_MUTED, font_size="0.76rem", min_width="120px"),
                            rx.text(row["preview_columns"], color=TEXT_MUTED, font_size="0.73rem", overflow="hidden", text_overflow="ellipsis", white_space="nowrap"),
                            width="100%",
                            padding="0.45rem 0.5rem",
                            border_bottom=f"1px solid rgba(30,45,71,0.5)",
                            _hover={"background": "rgba(79,140,255,0.04)"},
                        ),
                    ),
                    width="100%",
                    spacing="0",
                ),
                rx.text("No datasets in Data Lake yet.", color=TEXT_MUTED, font_size="0.85rem", padding="1rem 0"),
            ),
        ),
        # Recent runs
        card(
            section_title("Recent Experiment Runs", "Latest MLflow tracked runs."),
            rx.cond(
                AppState.runs.length() > 0,
                rx.vstack(
                    rx.hstack(
                        rx.text("EXPERIMENT", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px", min_width="150px"),
                        rx.text("STATUS", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px", min_width="90px"),
                        rx.text("STARTED", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px", min_width="130px"),
                        rx.text("METRICS", color=TEXT_MUTED, font_size="0.68rem", font_weight="700", letter_spacing="0.8px"),
                        width="100%",
                        padding="0.4rem 0.5rem",
                        border_bottom=f"1px solid {DARK_BORDER}",
                    ),
                    rx.foreach(
                        AppState.runs,
                        lambda row: rx.hstack(
                            rx.text(row["experiment"], color=TEXT_PRIMARY, font_weight="600", font_size="0.82rem", min_width="150px"),
                            rx.box(
                                row["status"],
                                padding="0.1rem 0.5rem", border_radius="999px", font_size="0.70rem", font_weight="700",
                                color=rx.cond(row["status"] == "FINISHED", ACCENT_2, rx.cond(row["status"] == "RUNNING", ACCENT, ACCENT_RED)),
                                background=rx.cond(row["status"] == "FINISHED", "rgba(50,201,168,0.12)", rx.cond(row["status"] == "RUNNING", "rgba(79,140,255,0.12)", "rgba(240,79,95,0.12)")),
                                min_width="90px",
                            ),
                            rx.text(row["started"], color=TEXT_MUTED, font_size="0.76rem", min_width="130px"),
                            rx.text(row["metrics"], color=TEXT_MUTED, font_size="0.74rem", overflow="hidden", text_overflow="ellipsis", white_space="nowrap"),
                            width="100%",
                            padding="0.45rem 0.5rem",
                            border_bottom=f"1px solid rgba(30,45,71,0.5)",
                            _hover={"background": "rgba(79,140,255,0.04)"},
                        ),
                    ),
                    width="100%",
                    spacing="0",
                ),
                rx.text("No experiment runs yet.", color=TEXT_MUTED, font_size="0.85rem", padding="1rem 0"),
            ),
        ),
        spacing="4",
        width="100%",
    )


def data_page() -> rx.Component:
    return rx.vstack(
        card(
            rx.text("Data Ingestion", color=TEXT_PRIMARY, font_weight="700"),
            rx.text("Attach one or more files (CSV, JSON, Parquet, TXT, ZIP) to save in the Data Lake.", color=TEXT_MUTED, font_size="0.82rem"),
            rx.upload(
                rx.vstack(
                    rx.box(
                        rx.icon("cloud_upload", size=24, color=ACCENT),
                        background="rgba(79,140,255,0.12)",
                        padding="0.8rem",
                        border_radius="12px",
                    ),
                    rx.button("Select File(s)", background=ACCENT_2, color="#041a17", font_weight="700"),
                    rx.text("Drag and drop files here (CSV, JSON, Parquet, TXT, ZIP)", color=TEXT_MUTED, font_size="0.8rem"),
                    spacing="2",
                    align_items="center",
                    width="100%",
                ),
                id="data_upload",
                multiple=True,
                width="100%",
                border=f"1px dashed {DARK_BORDER}",
                padding="2rem",
                border_radius="14px",
                _hover={"border_color": ACCENT, "background": "rgba(79,140,255,0.02)"},
            ),
            rx.accordion.root(
                rx.accordion.item(
                    rx.accordion.header(
                        rx.hstack(
                            rx.icon("settings-2", size=14),
                            rx.text("Advanced Parsing Options", font_size="0.85rem", font_weight="600"),
                            rx.spacer(),
                            rx.accordion.icon(),
                            width="100%",
                        ),
                    ),
                    rx.accordion.content(
                        rx.grid(
                            input_block("File Format", rx.select(["Auto", "CSV", "JSON", "Parquet", "TXT"], value="Auto", width="100%")),
                            input_block("Delimiter", rx.select([",", ";", "|", "tab"], value=",", width="100%")),
                            input_block("Encoding", rx.select(["utf-8", "latin-1", "ascii"], value="utf-8", width="100%")),
                            columns="3",
                            spacing="3",
                            width="100%",
                            padding_top="0.5rem",
                        ),
                    ),
                    value="advanced",
                ),
                width="100%",
                margin_top="0.8rem",
                variant="ghost",
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
            rx.hstack(
                rx.text("Schema & Split Controls", color=TEXT_PRIMARY, font_weight="700"),
                rx.spacer(),
                rx.button(
                    rx.hstack(rx.icon("table", size=12), rx.text("Load Schema"), spacing="1"),
                    on_click=AppState.load_schema_for_selected_dataset,
                    background=f"linear-gradient(90deg, {ACCENT}, #6aa3ff)",
                    color="#05112a", font_weight="700", size="1", border_radius="7px",
                ),
                width="100%", align_items="center",
            ),
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
            # Schema Override interactive table
            rx.cond(
                AppState.schema_rows.length() > 0,
                rx.vstack(
                    rx.text("Column Schema Override", color=TEXT_PRIMARY, font_size="0.85rem", font_weight="700", margin_top="0.8rem"),
                    rx.box(
                        # Header row
                        rx.hstack(
                            rx.text("COLUMN", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px", min_width="120px"),
                            rx.text("DTYPE", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px", min_width="90px"),
                            rx.text("OVERRIDE TYPE", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px", min_width="130px"),
                            rx.text("INCLUDE", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px"),
                            width="100%", padding="0.35rem 0.5rem",
                            border_bottom=f"1px solid {DARK_BORDER}",
                        ),
                        rx.foreach(
                            AppState.schema_rows,
                            lambda row: rx.hstack(
                                rx.hstack(
                                    rx.icon("table_columns_split", size=12, color=ACCENT),
                                    rx.text(row["column"], color=TEXT_PRIMARY, font_size="0.80rem", font_weight="600"),
                                    spacing="1", min_width="120px",
                                ),
                                rx.text(row["dtype"], color=TEXT_MUTED, font_size="0.74rem", min_width="90px"),
                                rx.select(
                                    ["Auto", "Categorical", "Numerical", "Text", "Datetime"],
                                    value=row["override_type"],
                                    on_change=lambda t: AppState.set_schema_type(row["column"], t),
                                    size="1",
                                    width="130px",
                                ),
                                rx.checkbox(
                                    "",
                                    is_checked=(row["include"] == "true"),
                                    on_change=lambda _: AppState.toggle_schema_include(row["column"]),
                                ),
                                width="100%", padding="0.35rem 0.5rem",
                                border_bottom=f"1px solid rgba(30,45,71,0.4)",
                                _hover={"background": "rgba(79,140,255,0.04)"},
                            ),
                        ),
                        border=f"1px solid {DARK_BORDER}",
                        border_radius="10px",
                        overflow="hidden",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text("Generated JSON Override", color=TEXT_MUTED, font_size="0.75rem", font_weight="500", margin_top="0.6rem"),
                        rx.code_block(AppState.data_schema_overrides_json, language="json", width="100%"),
                        width="100%", spacing="1", align_items="start",
                    ),
                    width="100%", spacing="2", align_items="start",
                ),
                rx.vstack(
                    rx.text("No schema loaded. Select a dataset & version above, then click Load Schema.", color=TEXT_MUTED, font_size="0.80rem"),
                    input_block(
                        "Manual JSON override (optional)",
                        rx.text_area(
                            value=AppState.data_schema_overrides_json,
                            on_change=AppState.set_data_schema_overrides_json,
                            min_height="80px",
                            width="100%",
                        ),
                    ),
                    width="100%", spacing="2", align_items="start",
                ),
            ),
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


# ---------------------------------------------------------------------------
# AutoML Step Wizard helpers
# ---------------------------------------------------------------------------

def _wizard_step_btn(step_num: int, label_text: str) -> rx.Component:
    is_active = AppState.automl_wizard_step == step_num
    is_done = AppState.automl_wizard_step > step_num
    return rx.button(
        rx.hstack(
            rx.box(
                str(step_num),
                width="22px", height="22px",
                border_radius="999px",
                display="flex",
                align_items="center",
                justify_content="center",
                font_size="0.70rem",
                font_weight="800",
                background=rx.cond(is_active, f"linear-gradient(90deg, {ACCENT}, {ACCENT_2})",
                                   rx.cond(is_done, ACCENT_2, "#253048")),
                color=rx.cond(is_active, "#04122d", rx.cond(is_done, "#041a17", TEXT_MUTED)),
                flex_shrink="0",
            ),
            rx.text(
                label_text,
                font_size="0.74rem",
                font_weight=rx.cond(is_active, "700", "500"),
                color=rx.cond(is_active, TEXT_PRIMARY, rx.cond(is_done, ACCENT_2, TEXT_MUTED)),
            ),
            spacing="2", align_items="center",
        ),
        on_click=lambda: AppState.wizard_goto(step_num),
        background="transparent",
        padding="0.4rem 0.6rem",
        border_radius="8px",
        border=rx.cond(is_active, f"1px solid rgba(79,140,255,0.5)", "1px solid transparent"),
        cursor="pointer",
        _hover={"background": "rgba(79,140,255,0.08)"},
    )


def wizard_nav_bar() -> rx.Component:
    steps = [(1, "Dataset & Task"), (2, "Model Source"), (3, "Optimization"), (4, "Validation"), (5, "Review & Submit")]
    return card(
        rx.hstack(
            *[_wizard_step_btn(n, lbl) for n, lbl in steps],
            rx.spacer(),
            rx.text(AppState.automl_wizard_step, "/5", color=TEXT_MUTED, font_size="0.75rem"),
            width="100%",
            align_items="center",
            overflow_x="auto",
        ),
        padding="0.65rem 0.9rem",
    )


def wizard_footer_btns() -> rx.Component:
    return rx.hstack(
        rx.button(
            rx.hstack(rx.icon("chevron-left", size=13), rx.text("Back"), spacing="1"),
            on_click=AppState.wizard_back,
            background="#253048",
            color=TEXT_PRIMARY,
            border_radius="8px",
            size="2",
            display=rx.cond(AppState.automl_wizard_step > 1, "flex", "none"),
        ),
        rx.spacer(),
        rx.cond(
            AppState.automl_wizard_step < 5,
            rx.button(
                rx.hstack(rx.text("Next"), rx.icon("chevron-right", size=13), spacing="1"),
                on_click=AppState.wizard_next,
                background=f"linear-gradient(90deg, {ACCENT}, #6aa3ff)",
                color="#05112a", font_weight="700", border_radius="8px", size="2",
            ),
            rx.button(
                rx.hstack(rx.icon("play", size=13), rx.text("Submit AutoML Job"), spacing="1"),
                on_click=AppState.submit_automl,
                background=f"linear-gradient(90deg, {ACCENT_2}, #26ddb5)",
                color="#04170f", font_weight="700", border_radius="8px", size="2",
            ),
        ),
        width="100%",
        margin_top="0.7rem",
    )


def automl_page() -> rx.Component:
    return rx.vstack(
        hero_header("AutoML Training Wizard", "Step-by-step guided training configuration."),
        wizard_nav_bar(),

        # ---- Step 1: Dataset & Task ----------------------------------------
        rx.cond(
            AppState.automl_wizard_step == 1,
            card(
                section_title("Step 1 - Dataset & Task", "Choose task type, dataset, and the target column."),
                rx.grid(
                    input_block(
                        "Task type",
                        rx.select(
                            ["classification", "regression", "clustering", "time_series", "anomaly_detection"],
                            value=AppState.automl_task,
                            on_change=AppState.update_automl_task,
                            width="100%",
                        ),
                    ),
                    input_block("Target column", rx.select(AppState.automl_target_options, value=AppState.automl_target_column, on_change=AppState.set_automl_target_column, width="100%")),
                    input_block("Experiment name", rx.input(value=AppState.automl_experiment_name, on_change=AppState.set_automl_experiment_name, width="100%")),
                    columns="2", spacing="3", width="100%",
                ),
                rx.grid(
                    input_block("Train dataset", rx.select(AppState.dataset_options, value=AppState.automl_train_dataset, on_change=AppState.update_automl_train_dataset, width="100%")),
                    input_block("Train version", rx.select(AppState.automl_train_version_options, value=AppState.automl_train_version, on_change=AppState.update_automl_train_version, width="100%")),
                    input_block("Test dataset (optional)", rx.select(AppState.dataset_options_optional, placeholder="— none —", value=AppState.automl_test_dataset, on_change=AppState.update_automl_test_dataset, width="100%")),
                    input_block("Test version (optional)", rx.select(AppState.automl_test_version_options_optional, placeholder="— none —", value=AppState.automl_test_version, on_change=AppState.set_automl_test_version, width="100%")),
                    columns="2", spacing="3", width="100%",
                ),
                wizard_footer_btns(),
            ),
        ),

        # ---- Step 2: Model Source & Focus ----------------------------------
        rx.cond(
            AppState.automl_wizard_step == 2,
            card(
                section_title("Step 2 - Model Source & Focus", "Choose what kind of model to train."),
                rx.grid(
                    input_block(
                        "Model source",
                        rx.select(
                            ["Standard AutoML", "Model Registry", "Local Upload"],
                            value=AppState.automl_model_source,
                            on_change=AppState.update_model_source,
                            width="100%",
                        ),
                    ),
                    input_block(
                        "Training focus",
                        rx.select(["single", "ensemble_only", "both"], value=AppState.automl_ensemble_mode, on_change=AppState.update_ensemble_mode, width="100%"),
                    ),
                    input_block(
                        "Model selection mode",
                        rx.select(["Automatic (Preset)", "Manual (Select)"], value=AppState.automl_mode_selection, on_change=AppState.update_mode_selection, width="100%"),
                    ),
                    input_block(
                        "Training strategy",
                        rx.select(["Automatic", "Manual"], value=AppState.automl_training_strategy, on_change=AppState.update_training_strategy, width="100%"),
                    ),
                    columns="2", spacing="3", width="100%",
                ),
                rx.checkbox("Include Deep Learning Models", is_checked=AppState.automl_use_deep_learning, on_change=AppState.update_use_deep_learning, margin_top="0.5rem"),
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
                                spacing="2", align_items="center", width="100%",
                            ),
                            id="model_upload", multiple=False, width="100%",
                            border=f"1px dashed {DARK_BORDER}", padding="1rem", border_radius="10px",
                        ),
                        rx.button("Upload model", on_click=AppState.handle_model_upload(rx.upload_files(upload_id="model_upload")), background=ACCENT, color="#06122d"),
                        rx.cond(AppState.automl_uploaded_model_status != "", rx.text(AppState.automl_uploaded_model_status, color=ACCENT_2, font_size="0.82rem")),
                        spacing="2", width="100%", align_items="start",
                    ),
                ),
                rx.cond(
                    AppState.automl_mode_selection == "Manual (Select)",
                    card(
                        section_title("Base Model Picker", "Select individual models to include."),
                        rx.flex(
                            rx.foreach(
                                AppState.automl_available_models,
                                lambda model_name: rx.button(
                                    model_name,
                                    on_click=lambda: AppState.toggle_automl_model(model_name),
                                    border_radius="999px", padding="0.35rem 0.75rem", font_size="0.78rem",
                                    background=rx.cond(AppState.automl_selected_model_list.contains(model_name), f"linear-gradient(90deg, rgba(79,140,255,0.95), rgba(50,201,168,0.8))", "#1b2539"),
                                    color=rx.cond(AppState.automl_selected_model_list.contains(model_name), "#051325", TEXT_PRIMARY),
                                    border=rx.cond(AppState.automl_selected_model_list.contains(model_name), "1px solid rgba(50,201,168,0.6)", f"1px solid {DARK_BORDER}"),
                                    transition="all 160ms ease",
                                ),
                            ),
                            wrap="wrap", spacing="2", margin_top="0.85rem",
                        ),
                        input_block(
                            "Selected models (comma separated)",
                            rx.text_area(value=AppState.automl_models_csv, on_change=AppState.set_automl_models_csv_input, min_height="70px", width="100%"),
                        ),
                    ),
                ),
                rx.hstack(
                    rx.button("Refresh available models", on_click=AppState.refresh_models_for_task, background="#3b4357", color=TEXT_PRIMARY, size="1"),
                    spacing="2",
                ),
                wizard_footer_btns(),
            ),
        ),

        # ---- Step 3: Optimization ------------------------------------------
        rx.cond(
            AppState.automl_wizard_step == 3,
            card(
                section_title("Step 3 - Optimization", "Configure preset, trials, budget, and metric."),
                rx.grid(
                    input_block("Preset", rx.select(["test", "fast", "medium", "high", "custom"], value=AppState.automl_preset, on_change=AppState.set_automl_preset, width="100%")),
                    input_block("Optimization mode", rx.select(["bayesian", "random", "grid", "hyperband"], value=AppState.automl_optimization_mode, on_change=AppState.set_automl_optimization_mode, width="100%")),
                    input_block(
                        "Optimization metric",
                        rx.select(AppState.automl_metric_options, value=AppState.automl_optimization_metric, on_change=AppState.set_automl_optimization_metric, width="100%"),
                    ),
                    input_block("Trials", rx.input(type="number", value=AppState.automl_n_trials, on_change=AppState.set_automl_n_trials_input, width="100%")),
                    input_block("Timeout (s)", rx.input(type="number", value=AppState.automl_timeout, on_change=AppState.set_automl_timeout_input, width="100%")),
                    input_block("Time budget (s)", rx.input(type="number", value=AppState.automl_time_budget, on_change=AppState.set_automl_time_budget_input, width="100%")),
                    columns="2", spacing="3", width="100%",
                ),
                card(
                    section_title("Custom Ensemble Builder", "Add custom voting, stacking, or bagging ensembles."),
                    rx.grid(
                        input_block("Ensemble type", rx.select(["Voting", "Stacking", "Bagging"], value=AppState.new_ensemble_type, on_change=AppState.set_new_ensemble_type, width="100%")),
                        input_block("Base models (comma separated)", rx.input(value=AppState.new_ensemble_models_csv, on_change=AppState.set_new_ensemble_models_csv, width="100%")),
                        input_block("Meta model (stacking)", rx.select(AppState.automl_available_models, value=AppState.new_ensemble_meta_model, on_change=AppState.set_new_ensemble_meta_model, width="100%")),
                        input_block("Voting type", rx.select(["soft", "hard"], value=AppState.new_ensemble_voting_type, on_change=AppState.set_new_ensemble_voting_type, width="100%")),
                        input_block("Bagging base", rx.select(AppState.automl_available_models, value=AppState.new_ensemble_bagging_base, on_change=AppState.set_new_ensemble_bagging_base, width="100%")),
                        columns="2", spacing="3", width="100%",
                    ),
                    rx.button("Add ensemble", on_click=AppState.add_custom_ensemble, background=ACCENT_2, color="#041a17"),
                    rx.foreach(
                        AppState.automl_custom_ensembles,
                        lambda item: card(
                            rx.hstack(
                                rx.text(item["type"], " | ", item["models"], " | meta: ", item["meta_model"], color=TEXT_MUTED, font_size="0.78rem"),
                                rx.spacer(),
                                rx.button("Remove", on_click=lambda: AppState.remove_custom_ensemble(item["id"]), background="#522b32", color=TEXT_PRIMARY, size="1"),
                                width="100%",
                            ),
                            padding="0.7rem",
                        ),
                    ),
                ),
                wizard_footer_btns(),
            ),
        ),

        # ---- Step 4: Validation --------------------------------------------
        rx.cond(
            AppState.automl_wizard_step == 4,
            card(
                section_title("Step 4 - Validation Strategy", "Configure cross-validation, holdout, and time-series settings."),
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
                    input_block("Folds", rx.input(type="number", value=AppState.automl_validation_folds, on_change=AppState.set_automl_validation_folds_input, width="100%")),
                    input_block("Holdout test size (%)", rx.input(type="number", value=AppState.automl_validation_test_size, on_change=AppState.set_automl_validation_test_size_input, width="100%")),
                    input_block("Time-series gap", rx.input(type="number", value=AppState.automl_validation_gap, on_change=AppState.set_automl_validation_gap_input, width="100%")),
                    input_block("Max train window (0=full)", rx.input(type="number", value=AppState.automl_validation_max_train_size, on_change=AppState.set_automl_validation_max_train_size_input, width="100%")),
                    input_block("Random seed", rx.input(type="number", value=AppState.automl_random_state, on_change=AppState.set_automl_random_state_input, width="100%")),
                    input_block("Early stopping", rx.input(type="number", value=AppState.automl_early_stopping, on_change=AppState.set_automl_early_stopping_input, width="100%")),
                    columns="2", spacing="3", width="100%",
                ),
                rx.hstack(
                    rx.checkbox("Shuffle", is_checked=AppState.automl_validation_shuffle, on_change=AppState.set_automl_validation_shuffle),
                    rx.checkbox("Stratify holdout", is_checked=AppState.automl_validation_stratify, on_change=AppState.set_automl_validation_stratify),
                    spacing="4", margin_top="0.55rem",
                ),
                rx.cond(
                    AppState.automl_training_strategy == "Manual",
                    rx.vstack(
                        rx.cond(AppState.automl_manual_param_cards != [], manual_param_cards()),
                        card(
                            rx.text("Manual Params Payload", color=TEXT_PRIMARY, font_weight="700"),
                            rx.code_block(AppState.automl_manual_params_json, language="json", width="100%"),
                        ),
                        width="100%", spacing="3", align_items="start",
                    ),
                ),
                wizard_footer_btns(),
            ),
        ),

        # ---- Step 5: Review & Submit ----------------------------------------
        rx.cond(
            AppState.automl_wizard_step == 5,
            rx.vstack(
                card(
                    section_title("Step 5 - Review & Submit", "Review your configuration before launching."),
                    rx.grid(
                        card(
                            rx.text("Task", color=TEXT_MUTED, font_size="0.72rem"), rx.text(AppState.automl_task, color=TEXT_PRIMARY, font_weight="700"),
                            padding="0.7rem",
                        ),
                        card(
                            rx.text("Target", color=TEXT_MUTED, font_size="0.72rem"), rx.text(AppState.automl_target_column, color=TEXT_PRIMARY, font_weight="700"),
                            padding="0.7rem",
                        ),
                        card(
                            rx.text("Dataset", color=TEXT_MUTED, font_size="0.72rem"), rx.text(AppState.automl_train_dataset, color=TEXT_PRIMARY, font_weight="700"),
                            padding="0.7rem",
                        ),
                        card(
                            rx.text("Preset", color=TEXT_MUTED, font_size="0.72rem"), rx.text(AppState.automl_preset, color=TEXT_PRIMARY, font_weight="700"),
                            padding="0.7rem",
                        ),
                        card(
                            rx.text("Metric", color=TEXT_MUTED, font_size="0.72rem"), rx.text(AppState.automl_optimization_metric, color=TEXT_PRIMARY, font_weight="700"),
                            padding="0.7rem",
                        ),
                        card(
                            rx.text("Validation", color=TEXT_MUTED, font_size="0.72rem"), rx.text(AppState.automl_validation_strategy, color=TEXT_PRIMARY, font_weight="700"),
                            padding="0.7rem",
                        ),
                        columns="3", spacing="3", width="100%",
                    ),
                    rx.text(AppState.automl_submit_status, color=ACCENT_2, font_size="0.85rem"),
                    wizard_footer_btns(),
                ),
                width="100%", spacing="3",
            ),
        ),

        spacing="4",
        width="100%",
    )


def experiments_page() -> rx.Component:
    return rx.vstack(
        hero_header("Experiments", "Monitor training jobs, view logs, and analyze results."),
        # Training results chart
        card(
            section_title("Training Results", "Interactive comparison and trial-progress charts for the selected job."),
            rx.cond(
                AppState.job_comparison_chart_json != "{}",
                rx.grid(
                    rx.vstack(
                        rx.text("Model Score Comparison", color=TEXT_MUTED, font_size="0.74rem", font_weight="600"),
                        rx.box(
                            # rx.plotly(data=AppState.job_comparison_chart_json, width="100%"),
                            rx.text("Chart data loading...", color=TEXT_MUTED),
                            width="100%",
                        ),
                        spacing="1", width="100%",
                    ),
                    rx.vstack(
                        rx.text("Trial Optimization Progress", color=TEXT_MUTED, font_size="0.74rem", font_weight="600"),
                        rx.cond(
                            AppState.job_progress_chart_json != "{}",
                            rx.box(
                                # rx.plotly(data=AppState.job_progress_chart_json, width="100%"),
                                rx.text("Progress data loading...", color=TEXT_MUTED),
                                width="100%",
                            ),
                            rx.vstack(
                                rx.icon("trending-up", size=28, color=TEXT_MUTED),
                                rx.text("Not enough trials to show progress.", color=TEXT_MUTED, font_size="0.80rem"),
                                align_items="center", justify_content="center", padding="2rem", width="100%",
                            ),
                        ),
                        spacing="1", width="100%",
                    ),
                    columns="2", spacing="4", width="100%",
                ),
                rx.cond(
                    AppState.job_chart_rows.length() > 0,
                    rx.vstack(
                        rx.foreach(
                            AppState.job_chart_rows,
                            lambda row: rx.vstack(
                                rx.hstack(
                                    rx.text(row["name"], color=TEXT_PRIMARY, font_size="0.82rem", font_weight="600"),
                                    rx.spacer(),
                                    rx.text(row["score"], color=ACCENT_2, font_size="0.82rem", font_weight="700"),
                                    width="100%",
                                ),
                                rx.box(
                                    rx.box(
                                        height="6px",
                                        width=row["width"],
                                        background=f"linear-gradient(90deg, {ACCENT}, {ACCENT_2})",
                                        border_radius="999px",
                                        transition="width 600ms ease",
                                    ),
                                    height="6px",
                                    width="100%",
                                    background=DARK_BORDER,
                                    border_radius="999px",
                                ),
                                spacing="1", width="100%", align_items="start",
                            ),
                        ),
                        width="100%", spacing="3",
                    ),
                    rx.text("No completed jobs with scores yet.", color=TEXT_MUTED, font_size="0.85rem", padding="0.5rem 0"),
                ),
            ),
        ),
        # Job manager
        card(
            rx.hstack(
                section_title("Background Training Jobs", "Manage and monitor background training processes."),
                rx.spacer(),
                rx.button(
                    rx.hstack(rx.icon("refresh-cw", size=13), rx.text("Refresh"), spacing="1"),
                    on_click=AppState.refresh_jobs,
                    background=f"linear-gradient(90deg, {ACCENT}, #6aa3ff)",
                    color="#05112a",
                    font_weight="700",
                    border_radius="8px",
                    size="1",
                ),
                align_items="start",
                width="100%",
            ),
            # Job table
            rx.cond(
                AppState.jobs.length() > 0,
                rx.vstack(
                    rx.hstack(
                        rx.text("JOB ID", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="80px"),
                        rx.text("NAME", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="120px"),
                        rx.text("STATUS", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="90px"),
                        rx.text("BEST SCORE", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="90px"),
                        rx.text("DURATION", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px"),
                        width="100%",
                        padding="0.4rem 0.5rem",
                        border_bottom=f"1px solid {DARK_BORDER}",
                    ),
                    rx.foreach(
                        AppState.jobs,
                        lambda item: rx.hstack(
                            rx.text(item["job_id"], color=TEXT_MUTED, font_size="0.74rem", font_family="monospace", min_width="80px"),
                            rx.text(item["name"], color=TEXT_PRIMARY, font_weight="600", font_size="0.80rem", min_width="120px"),
                            rx.box(
                                item["status_label"],
                                padding="0.1rem 0.5rem", border_radius="999px", font_size="0.70rem", font_weight="700",
                                color=rx.cond(item["status"] == "JobStatus.COMPLETED", ACCENT_2,
                                             rx.cond(item["status"] == "JobStatus.RUNNING", ACCENT, TEXT_MUTED)),
                                background=rx.cond(item["status"] == "JobStatus.COMPLETED", "rgba(50,201,168,0.12)",
                                                  rx.cond(item["status"] == "JobStatus.RUNNING", "rgba(79,140,255,0.12)", "rgba(255,255,255,0.05)")),
                                min_width="90px",
                            ),
                            rx.text(item["best_score"], color=ACCENT_2, font_weight="700", font_size="0.80rem", min_width="90px"),
                            rx.text(item["duration"], color=TEXT_MUTED, font_size="0.76rem"),
                            width="100%",
                            padding="0.45rem 0.5rem",
                            border_bottom=f"1px solid rgba(30,45,71,0.4)",
                            _hover={"background": "rgba(79,140,255,0.04)"},
                        ),
                    ),
                    width="100%",
                    spacing="0",
                ),
                rx.text("No training jobs submitted yet.", color=TEXT_MUTED, font_size="0.85rem", padding="0.5rem 0"),
            ),
            rx.hstack(
                input_block(
                    "Select Job",
                    rx.select(AppState.job_id_options, value=AppState.selected_job_id, on_change=AppState.update_selected_job, width="240px"),
                ),
                rx.spacer(),
                rx.vstack(
                    rx.text("Actions", color=TEXT_MUTED, font_size="0.78rem", font_weight="500"),
                    rx.hstack(
                        rx.button("Pause", on_click=AppState.pause_selected_job, background="rgba(255,184,77,0.12)", color=ACCENT_3, border=f"1px solid rgba(255,184,77,0.25)", border_radius="7px", size="1"),
                        rx.button("Resume", on_click=AppState.resume_selected_job, background="rgba(50,201,168,0.12)", color=ACCENT_2, border=f"1px solid rgba(50,201,168,0.25)", border_radius="7px", size="1"),
                        rx.button("Cancel", on_click=AppState.cancel_selected_job, background="rgba(240,79,95,0.12)", color=ACCENT_RED, border=f"1px solid rgba(240,79,95,0.25)", border_radius="7px", size="1"),
                        rx.button("Delete", on_click=AppState.delete_selected_job, background="rgba(240,79,95,0.06)", color=ACCENT_RED, border=f"1px solid rgba(240,79,95,0.15)", border_radius="7px", size="1"),
                        spacing="2",
                    ),
                    spacing="1",
                    align_items="start",
                ),
                width="100%",
                align_items="end",
                margin_top="0.8rem",
            ),
        ),
        # Job Detail Dashboard
        rx.cond(
            AppState.selected_job_id != "",
            card(
                rx.hstack(
                    rx.icon("bar-chart-2", size=15, color=ACCENT),
                    rx.text("Job Details", color=TEXT_PRIMARY, font_weight="700"),
                    rx.text(AppState.selected_job_id, color=TEXT_MUTED, font_size="0.72rem", font_family="monospace"),
                    spacing="2", align_items="center", width="100%",
                ),
                # Tab bar
                rx.hstack(
                    rx.button(
                        rx.hstack(rx.icon("table-2", size=12), rx.text("Results"), spacing="1"),
                        on_click=lambda: AppState.set_job_detail_tab("results"),
                        background=rx.cond(AppState.job_detail_tab == "results", f"linear-gradient(90deg, {ACCENT}, #6aa3ff)", "#1b2539"),
                        color=rx.cond(AppState.job_detail_tab == "results", "#05112a", TEXT_MUTED),
                        border_radius="8px", size="1", font_weight="600",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("activity", size=12), rx.text("MLflow"), spacing="1"),
                        on_click=lambda: AppState.set_job_detail_tab("mlflow"),
                        background=rx.cond(AppState.job_detail_tab == "mlflow", f"linear-gradient(90deg, {ACCENT}, #6aa3ff)", "#1b2539"),
                        color=rx.cond(AppState.job_detail_tab == "mlflow", "#05112a", TEXT_MUTED),
                        border_radius="8px", size="1", font_weight="600",
                    ),
                    rx.button(
                        rx.hstack(rx.icon("upload-cloud", size=12), rx.text("Register & Deploy"), spacing="1"),
                        on_click=lambda: AppState.set_job_detail_tab("register"),
                        background=rx.cond(AppState.job_detail_tab == "register", f"linear-gradient(90deg, {ACCENT_2}, #26ddb5)", "#1b2539"),
                        color=rx.cond(AppState.job_detail_tab == "register", "#04170f", TEXT_MUTED),
                        border_radius="8px", size="1", font_weight="600",
                    ),
                    spacing="2", margin_top="0.6rem", margin_bottom="0.7rem",
                ),

                # Results tab
                rx.cond(
                    AppState.job_detail_tab == "results",
                    rx.vstack(
                        rx.cond(
                            AppState.job_model_results.length() > 0,
                            rx.vstack(
                                rx.hstack(
                                    rx.text("MODEL", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px", min_width="130px"),
                                    rx.text("METRIC", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px", min_width="110px"),
                                    rx.text("VALUE", color=TEXT_MUTED, font_size="0.63rem", font_weight="700", letter_spacing="0.8px"),
                                    width="100%", padding="0.3rem 0.5rem", border_bottom=f"1px solid {DARK_BORDER}",
                                ),
                                rx.foreach(
                                    AppState.job_model_results,
                                    lambda r: rx.hstack(
                                        rx.text(r["model"], color=TEXT_PRIMARY, font_size="0.79rem", font_weight="600", min_width="130px"),
                                        rx.text(r["metric"], color=TEXT_MUTED, font_size="0.76rem", min_width="110px"),
                                        rx.text(r["value"], color=ACCENT_2, font_size="0.76rem", font_weight="700"),
                                        width="100%", padding="0.32rem 0.5rem",
                                        border_bottom=f"1px solid rgba(30,45,71,0.35)",
                                        _hover={"background": "rgba(79,140,255,0.04)"},
                                    ),
                                ),
                                width="100%", spacing="0",
                            ),
                            rx.text("No per-model metrics available for this job.", color=TEXT_MUTED, font_size="0.82rem"),
                        ),
                        width="100%", align_items="start",
                    ),
                ),

                # MLflow tab
                rx.cond(
                    AppState.job_detail_tab == "mlflow",
                    rx.vstack(
                        rx.hstack(
                            rx.icon("link", size=13, color=ACCENT_2),
                            rx.text("MLflow Run ID:", color=TEXT_MUTED, font_size="0.82rem"),
                            rx.text(AppState.job_detail_mlflow_run_id, color=ACCENT_2, font_size="0.82rem", font_family="monospace"),
                            spacing="2", align_items="center",
                        ),
                        rx.cond(
                            AppState.run_details_json != "",
                            rx.vstack(
                                rx.text("Run Details", color=TEXT_MUTED, font_size="0.74rem", font_weight="500"),
                                rx.code_block(AppState.run_details_json, language="json", width="100%", max_height="300px", overflow_y="auto"),
                                spacing="1", width="100%", align_items="start",
                            ),
                        ),
                        width="100%", spacing="3", align_items="start",
                    ),
                ),

                # Register & Deploy tab
                rx.cond(
                    AppState.job_detail_tab == "register",
                    rx.grid(
                        rx.vstack(
                            rx.text("Register to MLflow Model Registry", color=TEXT_PRIMARY, font_size="0.84rem", font_weight="700"),
                            input_block("Model name", rx.input(
                                value=AppState.job_detail_register_name,
                                on_change=AppState.set_job_detail_register_name,
                                width="100%", placeholder="my_model_name",
                            )),
                            rx.button(
                                rx.hstack(rx.icon("package", size=13), rx.text("Register Model"), spacing="1"),
                                on_click=AppState.register_job_model,
                                background=f"linear-gradient(90deg, {ACCENT}, #6aa3ff)",
                                color="#05112a", font_weight="700", border_radius="8px", size="2",
                            ),
                            rx.cond(
                                AppState.job_detail_register_result != "",
                                rx.text(AppState.job_detail_register_result, color=ACCENT_2, font_size="0.82rem"),
                            ),
                            spacing="3", align_items="start", width="100%",
                        ),
                        rx.vstack(
                            rx.text("Push to Hugging Face Hub", color=TEXT_PRIMARY, font_size="0.84rem", font_weight="700"),
                            input_block("HF Repo ID", rx.input(
                                value=AppState.job_detail_hf_repo,
                                on_change=AppState.set_job_detail_hf_repo,
                                width="100%", placeholder="username/model-name",
                            )),
                            input_block("HF Token", rx.input(
                                value=AppState.job_detail_hf_token,
                                on_change=AppState.set_job_detail_hf_token,
                                type="password", width="100%",
                            )),
                            rx.button(
                                rx.hstack(rx.icon("rocket", size=13), rx.text("Push to HF"), spacing="1"),
                                on_click=AppState.push_job_model_to_hf,
                                background=f"linear-gradient(90deg, #ffb84d, #ff9a00)",
                                color="#1a0e00", font_weight="700", border_radius="8px", size="2",
                            ),
                            rx.cond(
                                AppState.job_detail_hf_result != "",
                                rx.text(AppState.job_detail_hf_result, color=ACCENT_3, font_size="0.82rem"),
                            ),
                            spacing="3", align_items="start", width="100%",
                        ),
                        columns="2", spacing="4", width="100%",
                    ),
                ),
            ),
        ),
        # Logs
        card(
            section_title("Training Logs", "Live output from the selected training job."),
            colored_log_view(AppState.selected_job_logs),
            rx.cond(
                AppState.selected_job_error != "",
                rx.box(
                    rx.hstack(
                        rx.icon("circle-x", size=14, color=ACCENT_RED),
                        rx.text("Error", color=ACCENT_RED, font_weight="700", font_size="0.82rem"),
                        spacing="1",
                        align_items="center",
                    ),
                    colored_log_view(AppState.selected_job_error),
                    margin_top="0.6rem",
                    width="100%",
                ),
            ),
        ),
        spacing="4",
        width="100%",
    )


def registry_deploy_page() -> rx.Component:
    CONSUMPTION_CODE = '''import joblib

# Load the pipeline (preprocessor + model)
pipeline = joblib.load("model.pkl")  # or path from registry

import pandas as pd
data = pd.DataFrame([{"feature1": 1.0, "feature2": "A"}])
predictions = pipeline.predict(data)
print(predictions)

# Or use the REST API:
import requests
resp = requests.post(
    "http://127.0.0.1:8000/predict",
    headers={"x-api-key": "YOUR_KEY"},
    json={"data": [[1.0, 0]]}
)
print(resp.json())
'''
    return rx.vstack(
        hero_header("Registry & Deploy", "Manage registered models, register runs, and deploy to production."),
        # Model Consumption Code
        card(
            section_title("Model Consumption Code", "How to use your trained model pipeline in production."),
            rx.box(
                rx.code_block(
                    AppState.consumption_code,
                    language="python",
                    width="100%",
                    show_line_numbers=True,
                    theme="nord",
                ),
                background="#050810",
                border=f"1px solid {DARK_BORDER}",
                border_radius="10px",
                overflow="auto",
                width="100%",
                on_mount=AppState.load_consumption_code,
            ),
            rx.text(
                "Download the .pkl file from MLflow artifacts or via the API, then use it as shown above.",
                color=TEXT_MUTED, font_size="0.78rem", margin_top="0.5rem",
            ),
        ),
        # Registry table
        card(
            section_title("Model Registry", "All registered models and their current stage."),
            rx.cond(
                AppState.registry_rows.length() > 0,
                rx.vstack(
                    rx.hstack(
                        rx.text("NAME", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="150px"),
                        rx.text("VER", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="50px"),
                        rx.text("STAGE", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px", min_width="100px"),
                        rx.text("DESCRIPTION", color=TEXT_MUTED, font_size="0.65rem", font_weight="700", letter_spacing="0.8px"),
                        width="100%",
                        padding="0.4rem 0.5rem",
                        border_bottom=f"1px solid {DARK_BORDER}",
                    ),
                    rx.foreach(
                        AppState.registry_rows,
                        lambda item: rx.hstack(
                            rx.hstack(
                                rx.icon("package", size=13, color="#a78bfa"),
                                rx.text(item["name"], color=TEXT_PRIMARY, font_weight="700", font_size="0.82rem"),
                                spacing="1",
                                min_width="150px",
                            ),
                            rx.text("v", item["version"], color=ACCENT_2, font_weight="700", font_size="0.82rem", min_width="50px"),
                            rx.box(
                                item["stage"],
                                padding="0.1rem 0.5rem", border_radius="999px", font_size="0.70rem", font_weight="700",
                                color=ACCENT, background="rgba(79,140,255,0.12)",
                                min_width="100px",
                            ),
                            rx.text(item["description"], color=TEXT_MUTED, font_size="0.75rem"),
                            width="100%",
                            padding="0.45rem 0.5rem",
                            border_bottom=f"1px solid rgba(30,45,71,0.4)",
                            _hover={"background": "rgba(79,140,255,0.04)"},
                        ),
                    ),
                    width="100%",
                    spacing="0",
                ),
                rx.text("No models registered yet.", color=TEXT_MUTED, font_size="0.85rem", padding="0.5rem 0"),
            ),
            rx.grid(
                input_block("Model name", rx.select(AppState.model_name_options, value=AppState.selected_model_name, on_change=AppState.update_selected_model_name, width="100%")),
                input_block("Model version (optional)", rx.select(AppState.selected_model_version_options_optional, placeholder="— none —", value=AppState.selected_model_version, on_change=AppState.set_selected_model_version, width="100%")),
                columns="2",
                spacing="3",
                width="100%",
                margin_top="0.8rem",
            ),
            rx.button(
                rx.hstack(rx.icon("search", size=13), rx.text("Load model details"), spacing="1"),
                on_click=AppState.load_model_details,
                background=f"linear-gradient(90deg, {ACCENT}, #6aa3ff)",
                color="#05112a", font_weight="700", border_radius="8px", size="1",
                margin_top="0.5rem",
            ),
            rx.code_block(AppState.model_details_json, language="json", width="100%", margin_top="0.5rem"),
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
        # CV Inference Playground - appears after training
        rx.cond(
            AppState.cv_trained_this_session,
            card(
                rx.hstack(
                    rx.icon("image", size=16, color=ACCENT_2),
                    rx.text("Inference Playground", color=TEXT_PRIMARY, font_weight="800", font_size="1rem"),
                    rx.box(
                        "Model Ready",
                        background="rgba(50,201,168,0.14)",
                        color=ACCENT_2,
                        font_size="0.68rem",
                        font_weight="700",
                        padding="0.2rem 0.6rem",
                        border_radius="999px",
                        border="1px solid rgba(50,201,168,0.35)",
                    ),
                    spacing="2", align_items="center", width="100%",
                ),
                rx.text(
                    "Upload an image to test your trained CV model with a live prediction.",
                    color=TEXT_MUTED, font_size="0.82rem", margin_top="0.2rem",
                ),
                rx.upload(
                    rx.vstack(
                        rx.box(
                            rx.icon("image-plus", size=22, color=ACCENT),
                            background="rgba(79,140,255,0.12)",
                            padding="0.8rem", border_radius="10px",
                        ),
                        rx.button(
                            "Choose Image",
                            background=f"linear-gradient(90deg, {ACCENT}, #6aa3ff)",
                            color="#05112a", font_weight="700", size="2",
                        ),
                        rx.text(
                            "PNG, JPG, WEBP — drop here or click to browse",
                            color=TEXT_MUTED, font_size="0.78rem",
                        ),
                        spacing="2", align_items="center", width="100%",
                    ),
                    id="cv_inference_upload",
                    multiple=False,
                    on_drop=AppState.handle_cv_inference_upload(rx.upload_files(upload_id="cv_inference_upload")),
                    width="100%",
                    border=f"2px dashed {DARK_BORDER}",
                    border_radius="12px",
                    padding="1.5rem",
                    margin_top="0.9rem",
                    background="rgba(15,20,32,0.5)",
                    _hover={"border_color": f"rgba(79,140,255,0.55)", "background": "rgba(79,140,255,0.04)"},
                    transition="all 200ms ease",
                ),
                rx.cond(
                    AppState.cv_inference_status != "",
                    rx.hstack(
                        rx.icon(
                            "check-circle",
                            size=14,
                            color=rx.cond(AppState.cv_inference_status == "Prediction complete.", ACCENT_2, ACCENT_3),
                        ),
                        rx.text(
                            AppState.cv_inference_status,
                            color=rx.cond(AppState.cv_inference_status == "Prediction complete.", ACCENT_2, ACCENT_3),
                            font_size="0.84rem", font_weight="600",
                        ),
                        spacing="2", align_items="center", margin_top="0.7rem",
                    ),
                ),
                rx.cond(
                    AppState.cv_inference_result_json != "{}",
                    rx.vstack(
                        rx.text("Prediction Result", color=TEXT_MUTED, font_size="0.76rem", font_weight="600", margin_top="0.8rem"),
                        rx.code_block(AppState.cv_inference_result_json, language="json", width="100%"),
                        spacing="1", width="100%", align_items="start",
                    ),
                ),
            ),
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
        rx.html(f"<style>{GLOBAL_CSS}</style>"),
        # ambient glow orbs (decorative)
        rx.box(
            position="fixed",
            top="-120px",
            right="-40px",
            width="380px",
            height="380px",
            border_radius="999px",
            background="radial-gradient(circle, rgba(79,140,255,0.18) 0%, rgba(79,140,255,0) 72%)",
            pointer_events="none",
            z_index="0",
        ),
        rx.box(
            position="fixed",
            bottom="-140px",
            left="22%",
            width="420px",
            height="420px",
            border_radius="999px",
            background="radial-gradient(circle, rgba(50,201,168,0.12) 0%, rgba(50,201,168,0) 72%)",
            pointer_events="none",
            z_index="0",
        ),
        rx.box(
            position="fixed",
            top="40%",
            right="-80px",
            width="280px",
            height="280px",
            border_radius="999px",
            background="radial-gradient(circle, rgba(167,139,250,0.10) 0%, rgba(167,139,250,0) 72%)",
            pointer_events="none",
            z_index="0",
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
            position="relative",
            z_index="1",
        ),
        display="flex",
        height="100vh",
        background=DARK_BG,
        overflow="hidden",
        on_mount=AppState.initialize,
    )
