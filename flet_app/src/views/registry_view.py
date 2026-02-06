import flet as ft

@ft.component
def RegistryView():
    return ft.Column(
        expand=True,
        controls=[
            ft.Text("Model Registry", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Manage production-ready models and versions.", color="onsurfacevariant"),
            ft.GridView(
                expand=1,
                runs_count=3,
                max_extent=300,
                child_aspect_ratio=1.5,
                spacing=10,
                run_spacing=10,
                controls=[
                    ft.Card(
                        content=ft.Container(
                            padding=15,
                            content=ft.Column([
                                ft.Row([ft.Icon("model_training"), ft.Text("SentimentModel", weight="bold")]),
                                ft.Text("Version: 2", size=12),
                                ft.Badge(content=ft.Text("Production"), bgcolor="green"),
                                ft.Row([
                                    ft.TextButton("Details"),
                                    ft.TextButton("Deploy", icon="rocket_launch"),
                                ], alignment=ft.MainAxisAlignment.END)
                            ])
                        )
                    ) for _ in range(5)
                ]
            )
        ]
    )
