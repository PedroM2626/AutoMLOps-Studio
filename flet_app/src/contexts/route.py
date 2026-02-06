import flet as ft
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class RouteContextValue:
    route: str
    navigate: Callable[[str], None]

RouteContext = ft.create_context(RouteContextValue)
