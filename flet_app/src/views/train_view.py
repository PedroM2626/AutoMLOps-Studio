import flet as ft
import sys
import os
import time

# Add parent directory to sys.path to import existing engines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from automl_engine import AutoMLDataProcessor, AutoMLTrainer

@ft.component
def TrainView():
    # Local states
    task_type, set_task_type = ft.use_state("classification")
    n_trials, set_n_trials = ft.use_state(10)
    is_training, set_is_training = ft.use_state(False)
    trials_data, set_trials_data = ft.use_state([])

    def run_training(e):
        set_is_training(True)
        set_trials_data([])
        ft.context.page.update()
        
        # Simulate training for now as we need real data
        for i in range(n_trials):
            time.sleep(0.5)
            new_trial = {
                "Trial": i + 1,
                "Model": "RandomForest" if i % 2 == 0 else "XGBoost",
                "Score": 0.85 + (i * 0.01),
                "Duration": 1.2
            }
            trials_data.append(new_trial)
            set_trials_data(list(trials_data))
            ft.context.page.update()
            
        set_is_training(False)
        ft.context.page.update()

    return ft.Column(
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE,
        controls=[
            ft.Text("Automated Training", size=24, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.Column([
                    ft.Text("Configuration", size=18, weight=ft.FontWeight.BOLD),
                    ft.RadioGroup(
                        content=ft.Row([
                            ft.Radio(value="classification", label="Classification"),
                            ft.Radio(value="regression", label="Regression"),
                        ]),
                        value=task_type,
                        on_change=lambda e: set_task_type(e.data)
                    ),
                    ft.Slider(
                        min=1, max=50, divisions=49, 
                        label="Trials: {value}", 
                        value=n_trials,
                        on_change=lambda e: set_n_trials(int(e.control.value))
                    ),
                    ft.Dropdown(
                        label="Target Column",
                        options=[ft.DropdownOption("target"), ft.DropdownOption("label")],
                    ),
                    ft.FilledButton(
                        "Run AutoML Pipeline", 
                        icon="play_arrow_rounded",
                        on_click=run_training,
                        disabled=is_training
                    ),
                ], expand=1),
                ft.Column([
                    ft.Text("Training Status", size=18, weight=ft.FontWeight.BOLD),
                    ft.ProgressBar(visible=is_training),
                    ft.Text("Ready to start" if not is_training else "Optimizing models...", italic=True),
                ], expand=1),
            ]),
            ft.Divider(),
            ft.Text("Trials Progress", size=18, weight=ft.FontWeight.BOLD),
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Trial")),
                    ft.DataColumn(ft.Text("Model")),
                    ft.DataColumn(ft.Text("Score")),
                    ft.DataColumn(ft.Text("Duration (s)")),
                ],
                rows=[
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(str(t["Trial"]))),
                        ft.DataCell(ft.Text(t["Model"])),
                        ft.DataCell(ft.Text(f"{t['Score']:.4f}")),
                        ft.DataCell(ft.Text(f"{t['Duration']:.2f}")),
                    ]) for t in trials_data
                ]
            )
        ]
    )
