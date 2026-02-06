import flet as ft
from contexts.route import RouteContext

@ft.component
def NavigationItem(icon: str, label: str, route: str, selected: bool):
    route_context = ft.use_context(RouteContext)
    
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(icon, color="green" if selected else "onsurfacevariant"),
                ft.Text(label, color="green" if selected else "onsurfacevariant", weight=ft.FontWeight.BOLD if selected else None),
            ],
            spacing=10,
        ),
        padding=ft.padding.all(12),
        border_radius=10,
        bgcolor="green,0.1" if selected else None,
        on_click=lambda _: route_context.navigate(route),
        ink=True,
    )

@ft.component
def Navigation():
    route_context = ft.use_context(RouteContext)
    current_route = route_context.route
    
    return ft.Column(
        controls=[
            ft.Text("MENU", size=12, weight=ft.FontWeight.W_500, color="onsurfacevariant"),
            NavigationItem("dataset_rounded", "Data Lake", "/", current_route == "/"),
            NavigationItem("auto_fix_high_rounded", "AutoML Train", "/train", current_route == "/train"),
            NavigationItem("analytics_rounded", "Experiments", "/experiments", current_route == "/experiments"),
            NavigationItem("image_search_rounded", "Computer Vision", "/cv", current_route == "/cv"),
            NavigationItem("monitor_heart_rounded", "Monitoring", "/monitoring", current_route == "/monitoring"),
            NavigationItem("storage_rounded", "Model Registry", "/registry", current_route == "/registry"),
            ft.Divider(height=20),
            ft.Text("SYSTEM", size=12, weight=ft.FontWeight.W_500, color="onsurfacevariant"),
            NavigationItem("settings_rounded", "Settings", "/settings", current_route == "/settings"),
        ],
        spacing=5,
        width=250,
    )
