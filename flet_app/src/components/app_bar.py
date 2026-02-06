import flet as ft
from contexts.theme import ThemeContext

@ft.component
def AppBar():
    theme = ft.use_context(ThemeContext)
    
    # Debugging the icon button issue
    icon_name = "dark_mode" if theme.mode == ft.ThemeMode.DARK else "light_mode"
    
    return ft.AppBar(
        leading=ft.Icon("rocket_launch_rounded", color="greenaccent700"),
        leading_width=40,
        title=ft.Text("AutoMLOps Studio", weight=ft.FontWeight.BOLD),
        center_title=False,
        bgcolor="surfacevariant",
        actions=[
            ft.Container(
                content=ft.Image(
                    src="icon.png", 
                    width=30, 
                    height=30, 
                    border_radius=ft.BorderRadius.all(15)
                ),
                padding=ft.padding.only(right=10),
            ),
            ft.IconButton(
                icon=icon_name,
                on_click=lambda _: theme.toggle_mode(),
            ),
            ft.PopupMenuButton(
                items=[
                    ft.PopupMenuItem("Settings", icon="settings"),
                    ft.PopupMenuItem("Documentation", icon="description"),
                    ft.PopupMenuItem("GitHub", icon="code"),
                ]
            ),
        ],
    )
