import flet as ft
import os
import time
import random

def main(page: ft.Page):
    page.title = "Flet AutoML Advanced Demo"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window_width = 800
    page.window_height = 900

    # --- Funções de Utilidade ---
    def toggle_theme(e):
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        theme_btn.icon = "brightness_2" if page.theme_mode == ft.ThemeMode.LIGHT else "wb_sunny"
        page.update()

    def check_system_status(e):
        try:
            mlruns_exists = os.path.exists("mlruns")
            models_exists = os.path.exists("models")
            status_chip.label.value = "Sistema: Online" if mlruns_exists else "Sistema: Configuração Pendente"
            status_chip.bgcolor = ft.Colors.GREEN_100 if mlruns_exists else ft.Colors.ORANGE_100
            status_chip.label.color = ft.Colors.GREEN_700 if mlruns_exists else ft.Colors.ORANGE_700
            
            detail_text.value = f"Diretórios encontrados:\n- mlruns: {'✅' if mlruns_exists else '❌'}\n- models: {'✅' if models_exists else '❌'}"
        except Exception as ex:
            detail_text.value = f"Erro na verificação: {str(ex)}"
        page.update()

    def simulate_training(e):
        train_btn.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        status_msg.value = "Iniciando treinamento simulado..."
        page.update()

        stages = ["Analisando Dados", "Pré-processamento", "Otimizando Hiperparâmetros", "Validando Modelo"]
        for i, stage in enumerate(stages):
            status_msg.value = f"Etapa {i+1}/4: {stage}..."
            for p in range(25):
                progress_bar.value += 0.01
                time.sleep(0.03) # Mais rápido para demonstração
                page.update()
        
        status_msg.value = "Treinamento concluído com sucesso! (Simulado)"
        train_btn.disabled = False
        progress_bar.visible = False
        
        # Adiciona um "resultado" aleatório
        results_list.controls.insert(0, ft.ListTile(
            leading=ft.Icon("bolt", color=ft.Colors.AMBER),
            title=ft.Text(f"Modelo {random.randint(1000, 9999)}"),
            subtitle=ft.Text(f"Acurácia: {random.uniform(0.85, 0.99):.4f}"),
            trailing=ft.Text(time.strftime("%H:%M:%S"))
        ))
        page.update()

    # --- Componentes das Abas ---

    # Aba 1: Dashboard / Status
    status_chip = ft.Chip(
        label=ft.Text("Status não verificado"),
        bgcolor=ft.Colors.GREY_100,
        on_click=check_system_status
    )
    detail_text = ft.Text("Clique no chip acima para validar o ambiente.", size=12)
    
    status_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Text("Verificação de Ambiente", weight=ft.FontWeight.BOLD),
                status_chip,
                detail_text
            ]),
            padding=15
        )
    )

    progress_bar = ft.ProgressBar(width=400, color="blue", visible=False)
    status_msg = ft.Text("Pronto para simular.", italic=True)
    train_btn = ft.FilledButton("Simular Treinamento AutoML", icon="play_arrow", on_click=simulate_training)
    results_list = ft.ListView(expand=True, spacing=10, height=300)

    dashboard_view = ft.Column([
        status_card,
        ft.Container(height=10),
        ft.Text("Pipeline de Treinamento", size=18, weight=ft.FontWeight.BOLD),
        train_btn,
        progress_bar,
        status_msg,
        ft.Divider(),
        ft.Text("Histórico de Simulações:", weight=ft.FontWeight.BOLD),
        results_list
    ], scroll=ft.ScrollMode.AUTO)

    # Aba 2: Visualização (Gráficos Similares)
    visualization_view = ft.Column([
        ft.Text("Análise de Performance", size=18, weight=ft.FontWeight.BOLD),
        ft.Text("Métricas de Acurácia por Modelo (Simulado)"),
        ft.Container(
            content=ft.Column([
                ft.Row([ft.Text("Random Forest: "), ft.ProgressBar(value=0.85, width=300, color="blue")]),
                ft.Row([ft.Text("XGBoost: "), ft.ProgressBar(value=0.92, width=300, color="green")]),
                ft.Row([ft.Text("LightGBM: "), ft.ProgressBar(value=0.89, width=300, color="orange")]),
                ft.Row([ft.Text("Neural Network: "), ft.ProgressBar(value=0.95, width=300, color="red")]),
            ]),
            padding=20,
            border=ft.Border(
                ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT),
                ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT),
                ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT),
                ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)
            ),
            border_radius=10
        ),
        ft.FilledButton("Atualizar Métricas", icon="refresh", on_click=lambda _: page.update())
    ])

    # Aba 3: Configurações
    settings_view = ft.Column([
        ft.Text("Configurações do Projeto", size=18, weight=ft.FontWeight.BOLD),
        ft.TextField(label="API Key", password=True, can_reveal_password=True),
        ft.TextField(label="MLFlow Tracking URI", value="./mlruns"),
        ft.Dropdown(
            label="Estratégia de Otimização",
            options=[
                ft.dropdown.Option("Bayesian"),
                ft.dropdown.Option("Random Search"),
                ft.dropdown.Option("Grid Search"),
            ],
            value="Bayesian"
        ),
        ft.Switch(label="Habilitar Logs Detalhados", value=True),
        ft.Text("Split de Validação (%)"),
        ft.Slider(min=0, max=100, divisions=10, label="{value}%"),
        ft.FilledButton("Salvar Configurações", icon="save", on_click=lambda _: page.show_snack_bar(ft.SnackBar(ft.Text("Configurações salvas!"))))
    ], scroll=ft.ScrollMode.AUTO)

    # --- Estrutura Principal ---
    
    # Cabeçalho Fixo
    theme_btn = ft.IconButton(icon="brightness_2", on_click=toggle_theme)
    header = ft.Row(
        [
            ft.Text("AutoML Suite Demo", size=24, weight=ft.FontWeight.BOLD),
            theme_btn
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN
    )

    # Sistema de Navegação (Substituindo Tabs por botões simples devido a incompatibilidade de versão)
    content_container = ft.Container(content=dashboard_view, expand=True)

    def navigate(e):
        if e.control.data == "dashboard":
            content_container.content = dashboard_view
        elif e.control.data == "visualization":
            content_container.content = visualization_view
        elif e.control.data == "settings":
            content_container.content = settings_view
        page.update()

    nav_row = ft.Row([
        ft.TextButton("Dashboard", icon="dashboard", on_click=navigate, data="dashboard"),
        ft.TextButton("Visualização", icon="bar_chart", on_click=navigate, data="visualization"),
        ft.TextButton("Configurações", icon="settings", on_click=navigate, data="settings"),
    ], alignment=ft.MainAxisAlignment.CENTER)

    # Montagem da Página
    page.add(
        header,
        ft.Divider(),
        nav_row,
        ft.Divider(),
        content_container
    )

if __name__ == "__main__":
    ft.run(main)
