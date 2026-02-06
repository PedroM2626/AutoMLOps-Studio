import flet as ft
import pandas as pd
import sys
import os

# Add parent directory to sys.path to import existing engines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from mlops_utils import DataLake

@ft.component
def DataView():
    datalake = DataLake()
    
    # Local states for UI
    datasets, set_datasets = ft.use_state(datalake.list_datasets())
    selected_ds, set_selected_ds = ft.use_state(None)
    versions, set_versions = ft.use_state([])
    selected_ver, set_selected_ver = ft.use_state(None)
    preview_data, set_preview_data = ft.use_state(None)
    dataset_name, set_dataset_name = ft.use_state("my_dataset")

    def handle_file_result(e):
        if e.files:
            try:
                file_path = e.files[0].path
                df = pd.read_csv(file_path)
                set_preview_data(df.head(10))
                ft.context.page.update()
            except Exception as ex:
                print(f"Error loading CSV: {ex}")

    file_picker = ft.FilePicker()
    file_picker.on_result = handle_file_result

    def on_mount():
        ft.context.page.overlay.append(file_picker)
        ft.context.page.update()

    ft.on_mounted(on_mount)

    def handle_pick_files(e):
        file_picker.pick_files(
            allowed_extensions=["csv"],
            allow_multiple=False
        )

    def load_dataset_versions(ds_name):
        set_selected_ds(ds_name)
        vers = datalake.list_versions(ds_name)
        set_versions(vers)
        ft.context.page.update()

    def load_specific_version(version):
        set_selected_ver(version)
        df = datalake.load_version(selected_ds, version)
        set_preview_data(df.head(10))
        ft.context.page.update()

    def save_to_datalake(e):
        if preview_data is not None:
            datalake.save_dataset(preview_data, dataset_name)
            set_datasets(datalake.list_datasets())
            ft.context.page.update()

    return ft.Column(
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE,
        controls=[
            ft.Text("Data Lake & Management", size=24, weight=ft.FontWeight.BOLD),
            ft.Row(
                controls=[
                    ft.Card(
                        content=ft.Container(
                            padding=20,
                            content=ft.Column([
                                ft.Text("Upload New Dataset", size=18, weight=ft.FontWeight.BOLD),
                                ft.ElevatedButton(
                                    "Select CSV File",
                                    icon="upload_file",
                                    on_click=handle_pick_files
                                ),
                                ft.TextField(
                                    label="Dataset Name", 
                                    value=dataset_name,
                                    on_change=lambda e: set_dataset_name(e.control.value)
                                ),
                                ft.FilledButton(
                                    "Save to Data Lake", 
                                    icon="save",
                                    on_click=save_to_datalake
                                ),
                            ])
                        ),
                        expand=1,
                    ),
                    ft.Card(
                        content=ft.Container(
                            padding=20,
                            content=ft.Column([
                                ft.Text("Explore Versions", size=18, weight=ft.FontWeight.BOLD),
                                ft.Dropdown(
                                    label="Select Dataset",
                                    options=[ft.DropdownOption(d) for d in datasets],
                                    on_text_change=lambda e: load_dataset_versions(e.control.value)
                                ),
                                ft.Dropdown(
                                    label="Select Version",
                                    options=[ft.DropdownOption(v) for v in versions],
                                    on_text_change=lambda e: load_specific_version(e.control.value)
                                ),
                                ft.FilledButton("Load Version", icon="refresh"),
                            ])
                        ),
                        expand=1,
                    ),
                ],
            ),
            ft.Divider(),
            ft.Text("Data Preview", size=18, weight=ft.FontWeight.BOLD),
            ft.Column(
                controls=[
                    ft.DataTable(
                        columns=[ft.DataColumn(ft.Text(col)) for col in (preview_data.columns if preview_data is not None else [])],
                        rows=[
                            ft.DataRow(cells=[ft.DataCell(ft.Text(str(val))) for val in row])
                            for _, row in (preview_data.iterrows() if preview_data is not None else [])
                        ],
                    ) if preview_data is not None else ft.Text("No data loaded yet.", italic=True)
                ],
                scroll=ft.ScrollMode.ALWAYS,
            )
        ]
    )
