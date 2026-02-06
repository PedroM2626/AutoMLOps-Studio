import flet as ft

@ft.component
def ExperimentsView():
    return ft.Column(
        expand=True,
        controls=[
            ft.Text("MLflow Experiments", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Track and compare model runs across experiments.", color="onsurfacevariant"),
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Run ID")),
                    ft.DataColumn(ft.Text("Experiment")),
                    ft.DataColumn(ft.Text("Model")),
                    ft.DataColumn(ft.Text("Accuracy")),
                    ft.DataColumn(ft.Text("Status")),
                ],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text("f123...89")),
                        ft.DataCell(ft.Text("automl_classification")),
                        ft.DataCell(ft.Text("XGBoost")),
                        ft.DataCell(ft.Text("0.9245")),
                        ft.DataCell(ft.Icon("check_circle", color="green")),
                    ])
                ]
            )
        ]
    )
