from __future__ import annotations

import reflex as rx

from .components import shell
from .state import AppState


def index() -> rx.Component:
    return shell()


app = rx.App()
app.add_page(index, route="/", title="AutoMLOps Studio Reflex", on_load=AppState.initialize)