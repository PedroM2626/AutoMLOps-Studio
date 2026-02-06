import flet as ft

@ft.component
def CVView():
    def handle_image_result(e):
        if e.files:
            # Em uma implementação real, carregaríamos a imagem e rodaríamos inferência
            pass

    img_picker = ft.FilePicker()
    img_picker.on_result = handle_image_result

    def on_mount():
        ft.context.page.overlay.append(img_picker)
        ft.context.page.update()

    ft.on_mounted(on_mount)

    def handle_image_upload(e):
        img_picker.pick_files(
            allowed_extensions=["png", "jpg", "jpeg"]
        )

    return ft.Column(
        expand=True,
        controls=[
            ft.Text("Computer Vision AutoML", size=24, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.Card(
                    content=ft.Container(
                        padding=20,
                        content=ft.Column([
                            ft.Text("Image Classification", size=18, weight=ft.FontWeight.BOLD),
                            ft.TextField(label="Dataset Directory (folders by class)"),
                            ft.Row([
                                ft.Dropdown(
                                    label="Model Architecture",
                                    options=[
                                        ft.DropdownOption("ResNet50"),
                                        ft.DropdownOption("EfficientNetB0"),
                                        ft.DropdownOption("MobileNetV2"),
                                    ],
                                    expand=True,
                                ),
                                ft.TextField(label="Epochs", value="5", width=100),
                            ]),
                            ft.FilledButton("Start CV Training", icon="image_search"),
                        ])
                    ),
                    expand=1,
                ),
                ft.Card(
                    content=ft.Container(
                        padding=20,
                        content=ft.Column([
                            ft.Text("Inference Test", size=18, weight=ft.FontWeight.BOLD),
                            ft.ElevatedButton(
                                "Upload Image", 
                                icon="upload",
                                on_click=handle_image_upload
                            ),
                            ft.Image(src="https://via.placeholder.com/150", width=150, height=150),
                            ft.Text("Prediction: -", size=16),
                        ])
                    ),
                    expand=1,
                )
            ])
        ]
    )
